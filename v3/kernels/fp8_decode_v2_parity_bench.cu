// Standalone FP8 paged-decode parity and timing harness.
//
// Build: nvcc -O3 -arch=sm_90a --use_fast_math -o parity_bench parity_bench.cu
// Run:   ./parity_bench --json results.json
//
// Checks, in order:
//   0. Exhaustive fp8 dequant self-test: hardware cvt.rn.f16x2.e4m3x2
//      path vs fp8e4m3_to_float for all 256 encodings (value-exact;
//      NaN encodings must both be NaN; -0 vs +0 allowed).
//   1. Determinism: reference twice -> bitwise equal; v2 twice ->
//      bitwise equal (first shape point).
//   2. Parity per shape point under caller-supplied tolerances.
//   3. CUDA-event timing, reported without a promotion threshold.
//
// Data: random FP8 bytes with the two NaN encodings (0x7F/0xFF)
// remapped (real quantized KV never stores NaN); random per-slot
// descales in [0.002, 0.01] so f16 outputs land O(0.01..4) where the
// comparison remains in a numerically useful range; block tables are
// random permutations to exercise paging.

#include "fp8_decode_v2.cu"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#define CUDA_CHECK(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, \
                cudaGetErrorString(_e)); \
        exit(2); \
    } \
} while (0)

// Allocation failures are test failures; this harness never waits and retries.
static void* device_malloc(size_t bytes) {
    void* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, bytes));
    return p;
}

static float h2f(uint16_t h) {
    int s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF;
    float v;
    if (e == 31) v = m ? NAN : INFINITY;
    else if (e == 0) v = ldexpf((float)m, -24);
    else v = ldexpf((float)(1024 + m), e - 25);
    return s ? -v : v;
}

// ------------------------------------------------------------------
// 0. dequant self-test
// ------------------------------------------------------------------

__global__ void conv_test_kernel(float* sw, float* hw4) {
    int i = threadIdx.x; // 256 threads
    sw[i] = fp8e4m3_to_float((uint8_t)i);
    float f[4];
    fp8x4_to_float4((uint32_t)i * 0x01010101u, f); // byte i in all four lanes
    for (int j = 0; j < 4; j++) hw4[i * 4 + j] = f[j];
}

static bool run_conv_selftest() {
    float *d_sw = (float*)device_malloc(256 * 4);
    float *d_hw = (float*)device_malloc(1024 * 4);
    conv_test_kernel<<<1, 256>>>(d_sw, d_hw);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> sw(256), hw(1024);
    CUDA_CHECK(cudaMemcpy(sw.data(), d_sw, 256 * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hw.data(), d_hw, 1024 * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_sw)); CUDA_CHECK(cudaFree(d_hw));

    int bad = 0, zero_sign_diffs = 0;
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 4; j++) {
            float a = sw[i], b = hw[i * 4 + j];
            if (std::isnan(a) && std::isnan(b)) continue;
            if (a == b) { // value-equal; count ±0 bit diffs separately
                uint32_t ua, ub;
                memcpy(&ua, &a, 4); memcpy(&ub, &b, 4);
                if (ua != ub) zero_sign_diffs++;
                continue;
            }
            if (bad < 8)
                fprintf(stderr, "  conv mismatch byte=0x%02X lane=%d sw=%g hw=%g\n",
                        i, j, a, b);
            bad++;
        }
    }
    printf("[CONV ] hw cvt vs fp8e4m3_to_float over 256 encodings x 4 lanes: "
           "%s (%d mismatches, %d zero-sign diffs)\n",
           bad == 0 ? "PASS" : "FAIL", bad, zero_sign_diffs);
    return bad == 0;
}

// ------------------------------------------------------------------
// shape points
// ------------------------------------------------------------------

struct Point {
    const char* tag;       // hd256 / hd512
    int H, KVH, HD;
    int ctx, bs, batch;
    int ctx0;              // ctx of seq 0 when batch == 2
    bool null_qscale;
    bool null_kvscale;
    bool perf;             // include in perf table
};

static const Point POINTS[] = {
    // ---- hd256: sliding layers (32q / 16kv), runtime block_size 32 ----
    {"hd256", 32, 16, 256,    1, 32, 1, 0, false, false, false},
    {"hd256", 32, 16, 256,    2, 32, 1, 0, false, false, false},
    {"hd256", 32, 16, 256,   17, 32, 1, 0, false, false, true },
    {"hd256", 32, 16, 256,  128, 32, 1, 0, false, false, true },
    {"hd256", 32, 16, 256,  432, 32, 1, 0, false, false, true },
    {"hd256", 32, 16, 256, 1024, 32, 1, 0, false, false, true },
    // parity-only: block_size 1 and 16, batch 2 (seq0 ctx=0), scale fallbacks
    {"hd256", 32, 16, 256,  128,  1, 1, 0, false, false, false},
    {"hd256", 32, 16, 256, 1024,  1, 1, 0, false, false, false},
    {"hd256", 32, 16, 256,  128, 16, 1, 0, false, false, false},
    {"hd256", 32, 16, 256, 1024, 16, 1, 0, false, false, false},
    {"hd256", 32, 16, 256,  432, 32, 2, 0, false, false, false},
    {"hd256", 32, 16, 256,  128, 32, 1, 0, true,  false, false},
    {"hd256", 32, 16, 256,  128, 32, 1, 0, false, true,  false},
    // ---- hd512: Gemma 4 31B global layers (32q / 4kv) ----
    {"hd512-gqa8", 32, 4, 512,    1, 32, 1, 0, false, false, false},
    {"hd512-gqa8", 32, 4, 512,    2, 32, 1, 0, false, false, false},
    {"hd512-gqa8", 32, 4, 512,   17, 32, 1, 0, false, false, false},
    // Additional GQA4 coverage for the same head dimension.
    {"hd512", 16,  4, 512,  432, 32, 1, 0, false, false, true },
    {"hd512", 16,  4, 512, 1300, 32, 1, 0, false, false, true },
    {"hd512", 16,  4, 512, 8192, 32, 1, 0, false, false, true },
    {"hd512", 16,  4, 512, 1300,  1, 1, 0, false, false, false},
    {"hd512", 16,  4, 512, 1300, 16, 1, 0, false, false, false},
};

// Reference launcher.
static void launch_ref(const Point& P,
                       const uint8_t* q, const uint8_t* k, const uint8_t* v,
                       __half* out, const int* bt, const int* cl,
                       const float* ks, const float* vs, const float* qs,
                       const float* qd, const float* kd, const float* vd,
                       float scale, int mbps, int blocks_total) {
    dim3 grid(P.batch * P.H);
    if (P.HD == 256)
        paged_decode_fp8_kernel_reference<256><<<grid, 256>>>(
            q, k, v, out, bt, cl, ks, vs, qs, qd, kd, vd,
            scale, P.H, P.KVH, P.bs, mbps, blocks_total, -1);
    else if (P.HD == 512)
        paged_decode_fp8_kernel_reference<512><<<grid, 512>>>(
            q, k, v, out, bt, cl, ks, vs, qs, qd, kd, vd,
            scale, P.H, P.KVH, P.bs, mbps, blocks_total, -1);
    else
        paged_decode_fp8_kernel_reference<128><<<grid, 128>>>(
            q, k, v, out, bt, cl, ks, vs, qs, qd, kd, vd,
            scale, P.H, P.KVH, P.bs, mbps, blocks_total, -1);
}

template <typename F>
static double bench_us(F&& fn, int warmup, int iters) {
    for (int i = 0; i < warmup; i++) fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; i++) fn();
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    CUDA_CHECK(cudaEventDestroy(a)); CUDA_CHECK(cudaEventDestroy(b));
    return (double)ms * 1000.0 / iters;
}

struct Row {
    Point p;
    double max_abs, max_rel;
    bool pass;
    double ref_us, v2_us;
};

struct Config {
    uint32_t seed = 1234567u;
    double abs_tol = 2e-3;
    double rel_tol = 1e-2;
    int warmup = 100;
    int iters = 1000;
    const char* json_path = nullptr;
};

static bool parse_u32(const char* text, uint32_t* out) {
    char* end = nullptr;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno || end == text || *end || value > UINT32_MAX) return false;
    *out = (uint32_t)value;
    return true;
}

static bool parse_positive_int(const char* text, int* out) {
    char* end = nullptr;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno || end == text || *end || value <= 0 || value > INT_MAX) return false;
    *out = (int)value;
    return true;
}

static bool parse_positive_double(const char* text, double* out) {
    char* end = nullptr;
    errno = 0;
    double value = strtod(text, &end);
    if (errno || end == text || *end || !std::isfinite(value) || value <= 0.0) return false;
    *out = value;
    return true;
}

static Config parse_config(int argc, char** argv) {
    Config c;
    for (int i = 1; i < argc; i++) {
        if (i + 1 >= argc) {
            fprintf(stderr, "missing value for %s\n", argv[i]);
            exit(2);
        }
        const char* value = argv[++i];
        bool ok = false;
        if (!strcmp(argv[i - 1], "--seed")) ok = parse_u32(value, &c.seed);
        else if (!strcmp(argv[i - 1], "--abs-tol")) ok = parse_positive_double(value, &c.abs_tol);
        else if (!strcmp(argv[i - 1], "--rel-tol")) ok = parse_positive_double(value, &c.rel_tol);
        else if (!strcmp(argv[i - 1], "--warmup")) ok = parse_positive_int(value, &c.warmup);
        else if (!strcmp(argv[i - 1], "--iters")) ok = parse_positive_int(value, &c.iters);
        else if (!strcmp(argv[i - 1], "--json")) { c.json_path = value; ok = value[0] != '\0'; }
        else {
            fprintf(stderr, "unknown option: %s\n", argv[i - 1]);
            exit(2);
        }
        if (!ok) {
            fprintf(stderr, "invalid value for %s: %s\n", argv[i - 1], value);
            exit(2);
        }
    }
    if (c.json_path == nullptr) {
        fprintf(stderr, "--json NEW_PATH is required\n");
        exit(2);
    }
    return c;
}

static void json_string(FILE* f, const char* text) {
    fputc('"', f);
    for (const unsigned char* p = (const unsigned char*)text; *p; p++) {
        if (*p == '"' || *p == '\\') { fputc('\\', f); fputc(*p, f); }
        else if (*p >= 0x20) fputc(*p, f);
    }
    fputc('"', f);
}

int main(int argc, char** argv) {
    const Config config = parse_config(argc, argv);
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int driver_version = 0, runtime_version = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    size_t fre = 0, tot = 0;
    CUDA_CHECK(cudaMemGetInfo(&fre, &tot));
    printf("device: %s  sm_%d%d  free %.1f GiB / %.1f GiB\n",
           prop.name, prop.major, prop.minor,
           fre / 1073741824.0, tot / 1073741824.0);

    const bool conv_pass = run_conv_selftest();
    bool all_pass = conv_pass;

    size_t workspace_bytes = 0;
    for (const Point& P : POINTS) {
        const uint64_t queried = fa_sm89_decode_workspace_size(P.batch, P.H, P.KVH, P.HD);
        if (queried == UINT64_MAX || queried > SIZE_MAX) {
            fprintf(stderr, "invalid workspace query for %s\n", P.tag);
            exit(2);
        }
        workspace_bytes = std::max(workspace_bytes, static_cast<size_t>(queried));
    }
    float* d_ws = workspace_bytes ? (float*)device_malloc(workspace_bytes) : nullptr;

    std::vector<Row> rows;
    bool det_done = false;

    for (const Point& P : POINTS) {
        fprintf(stderr, "[point] %s ctx=%d bs=%d batch=%d qs0=%d kvs0=%d ... gen\n",
                P.tag, P.ctx, P.bs, P.batch, (int)P.null_qscale, (int)P.null_kvscale);
        std::mt19937 rng(config.seed + P.HD * 131u + P.ctx * 7u + P.bs * 3u + P.batch
                         + (P.null_qscale ? 17 : 0) + (P.null_kvscale ? 29 : 0));
        const int max_ctx = P.ctx;
        const int blocks_needed = (max_ctx + P.bs - 1) / P.bs;
        const int mbps = blocks_needed;
        const int blocks_total = blocks_needed + 2;
        const size_t slots = (size_t)blocks_total * P.bs;
        const size_t kv_bytes = slots * P.KVH * P.HD;
        const size_t q_bytes = (size_t)P.batch * P.H * P.HD;
        const size_t out_elems = (size_t)P.batch * P.H * P.HD;

        // host data
        auto rnd_fp8 = [&](std::vector<uint8_t>& v) {
            for (auto& b : v) {
                uint8_t x = (uint8_t)(rng() & 0xFF);
                if ((x & 0x7F) == 0x7F) x ^= 1; // remap NaN encodings
                b = x;
            }
        };
        std::vector<uint8_t> h_k(kv_bytes), h_v(kv_bytes), h_q(q_bytes);
        rnd_fp8(h_k); rnd_fp8(h_v); rnd_fp8(h_q);

        std::uniform_real_distribution<float> ds(0.002f, 0.01f);
        std::vector<float> h_ks(slots * P.KVH), h_vs(slots * P.KVH),
            h_qs((size_t)P.batch * P.H);
        for (auto& f : h_ks) f = ds(rng);
        for (auto& f : h_vs) f = ds(rng);
        for (auto& f : h_qs) f = ds(rng);

        std::vector<int> h_bt((size_t)P.batch * mbps);
        {
            std::vector<int> perm(blocks_total);
            for (int i = 0; i < blocks_total; i++) perm[i] = i;
            for (int b = 0; b < P.batch; b++) {
                std::shuffle(perm.begin(), perm.end(), rng);
                for (int i = 0; i < mbps; i++) h_bt[(size_t)b * mbps + i] = perm[i];
            }
        }
        std::vector<int> h_cl(P.batch);
        for (int b = 0; b < P.batch; b++)
            h_cl[b] = (P.batch == 2 && b == 0) ? P.ctx0 : P.ctx;

        // device data
        uint8_t* d_k = (uint8_t*)device_malloc(kv_bytes);
        uint8_t* d_v = (uint8_t*)device_malloc(kv_bytes);
        uint8_t* d_q = (uint8_t*)device_malloc(q_bytes);
        float* d_ksc = (float*)device_malloc(h_ks.size() * 4);
        float* d_vsc = (float*)device_malloc(h_vs.size() * 4);
        float* d_qsc = (float*)device_malloc(h_qs.size() * 4);
        int* d_bt = (int*)device_malloc(h_bt.size() * 4);
        int* d_cl = (int*)device_malloc(h_cl.size() * 4);
        float* d_fall = (float*)device_malloc(3 * 4); // q/k/v descale fallbacks
        __half* d_out_ref = (__half*)device_malloc(out_elems * 2);
        __half* d_out_v2 = (__half*)device_malloc(out_elems * 2);
        __half* d_out_tmp = (__half*)device_malloc(out_elems * 2);

        CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), kv_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), kv_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), q_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ksc, h_ks.data(), h_ks.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vsc, h_vs.data(), h_vs.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_qsc, h_qs.data(), h_qs.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bt, h_bt.data(), h_bt.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cl, h_cl.data(), h_cl.size() * 4, cudaMemcpyHostToDevice));
        float h_fall[3] = {0.004f, 0.006f, 0.005f};
        CUDA_CHECK(cudaMemcpy(d_fall, h_fall, 12, cudaMemcpyHostToDevice));

        const float scale = 1.0f / sqrtf((float)P.HD);
        const float* qs_arg = P.null_qscale ? nullptr : d_qsc;
        const float* ksc_arg = P.null_kvscale ? nullptr : d_ksc;
        const float* vsc_arg = P.null_kvscale ? nullptr : d_vsc;

        auto run_ref = [&](__half* out) {
            launch_ref(P, d_q, d_k, d_v, out, d_bt, d_cl, ksc_arg, vsc_arg,
                       qs_arg, d_fall + 0, d_fall + 1, d_fall + 2, scale, mbps,
                       blocks_total);
            CUDA_CHECK(cudaGetLastError());
        };
        auto run_v2 = [&](__half* out) {
            int rc = fa_sm89_paged_decode_fp8(
                (void*)d_q, (void*)d_k, (void*)d_v, (void*)out,
                (void*)d_bt, (void*)d_cl, (void*)d_ws,
                workspace_bytes,
                (void*)ksc_arg, (void*)vsc_arg, (void*)qs_arg,
                d_fall + 0, d_fall + 1, d_fall + 2,
                scale, P.batch, P.H, P.KVH, P.HD, P.bs, mbps,
                blocks_total, -1, nullptr);
            if (rc != 0) {
                fprintf(stderr, "FATAL: fa_sm89_paged_decode_fp8 rc=%d\n", rc);
                exit(2);
            }
        };

        CUDA_CHECK(cudaMemset(d_out_ref, 0x5A, out_elems * 2));
        CUDA_CHECK(cudaMemset(d_out_v2, 0x5A, out_elems * 2));
        fprintf(stderr, "  ... ref run\n");
        run_ref(d_out_ref);
        CUDA_CHECK(cudaDeviceSynchronize());
        fprintf(stderr, "  ... v2 run\n");
        run_v2(d_out_v2);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());

        // determinism (first point only)
        if (!det_done) {
            std::vector<uint16_t> a(out_elems), b(out_elems);
            run_ref(d_out_tmp);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(a.data(), d_out_ref, out_elems * 2, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(b.data(), d_out_tmp, out_elems * 2, cudaMemcpyDeviceToHost));
            bool ref_det = memcmp(a.data(), b.data(), out_elems * 2) == 0;
            run_v2(d_out_tmp);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(a.data(), d_out_v2, out_elems * 2, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(b.data(), d_out_tmp, out_elems * 2, cudaMemcpyDeviceToHost));
            bool v2_det = memcmp(a.data(), b.data(), out_elems * 2) == 0;
            printf("[DETRM] ref twice: %s, v2 twice: %s\n",
                   ref_det ? "bitwise-equal" : "MISMATCH",
                   v2_det ? "bitwise-equal" : "MISMATCH");
            if (!ref_det || !v2_det) all_pass = false;
            det_done = true;
        }

        // parity
        std::vector<uint16_t> h_ref(out_elems), h_new(out_elems);
        CUDA_CHECK(cudaMemcpy(h_ref.data(), d_out_ref, out_elems * 2, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_new.data(), d_out_v2, out_elems * 2, cudaMemcpyDeviceToHost));
        double max_abs = 0.0, max_rel = 0.0;
        long nan_count = 0;
        for (size_t i = 0; i < out_elems; i++) {
            float r = h2f(h_ref[i]), n = h2f(h_new[i]);
            if (std::isnan(r) || std::isnan(n)) { nan_count++; continue; }
            double ad = fabs((double)r - (double)n);
            if (ad > max_abs) max_abs = ad;
            if (fabs(r) > 0.1) {
                double rel = ad / fabs((double)r);
                if (rel > max_rel) max_rel = rel;
            }
        }
        bool pass = (nan_count == 0) && (max_abs <= config.abs_tol) &&
                    (max_rel <= config.rel_tol);
        if (!pass) all_pass = false;
        printf("[PARITY] %s ctx=%-5d bs=%-2d batch=%d%s%s%s : max_abs=%.3e max_rel=%.3e %s\n",
               P.tag, P.ctx, P.bs, P.batch,
               P.batch == 2 ? " (seq0 ctx=0)" : "",
               P.null_qscale ? " qs=fallback" : "",
               P.null_kvscale ? " kvs=fallback" : "",
               max_abs, max_rel, pass ? "PASS" : "FAIL");
        if (nan_count) printf("        !! %ld NaN outputs\n", nan_count);

        // bench
        double ref_us = 0.0, v2_us = 0.0;
        if (P.perf) {
            fprintf(stderr, "  ... bench ref\n");
            ref_us = bench_us([&] { run_ref(d_out_ref); }, config.warmup, config.iters);
            fprintf(stderr, "  ... bench v2\n");
            v2_us = bench_us([&] { run_v2(d_out_v2); }, config.warmup, config.iters);
            printf("[BENCH ] %s ctx=%-5d bs=%-2d : ref=%8.2f us  v2=%7.2f us  ratio=%5.2fx\n",
                   P.tag, P.ctx, P.bs, ref_us, v2_us, ref_us / v2_us);
        }

        rows.push_back({P, max_abs, max_rel, pass, ref_us, v2_us});

        CUDA_CHECK(cudaFree(d_k)); CUDA_CHECK(cudaFree(d_v)); CUDA_CHECK(cudaFree(d_q));
        CUDA_CHECK(cudaFree(d_ksc)); CUDA_CHECK(cudaFree(d_vsc)); CUDA_CHECK(cudaFree(d_qsc));
        CUDA_CHECK(cudaFree(d_bt)); CUDA_CHECK(cudaFree(d_cl)); CUDA_CHECK(cudaFree(d_fall));
        CUDA_CHECK(cudaFree(d_out_ref)); CUDA_CHECK(cudaFree(d_out_v2)); CUDA_CHECK(cudaFree(d_out_tmp));
    }
    CUDA_CHECK(cudaFree(d_ws));

    // markdown tables
    printf("\n## Parity\n\n");
    printf("| shape | ctx | block_size | batch | variant | max_abs | max_rel (|ref|>0.1) | gate |\n");
    printf("|---|---|---|---|---|---|---|---|\n");
    for (const Row& r : rows) {
        char variant[64] = "per-slot scales";
        if (r.p.null_qscale) snprintf(variant, sizeof variant, "q_descale fallback");
        if (r.p.null_kvscale) snprintf(variant, sizeof variant, "k/v descale fallback");
        if (r.p.batch == 2) snprintf(variant, sizeof variant, "batch2, seq0 ctx=0");
        printf("| %s | %d | %d | %d | %s | %.2e | %.2e | %s |\n",
               r.p.tag, r.p.ctx, r.p.bs, r.p.batch, variant,
               r.max_abs, r.max_rel, r.pass ? "PASS" : "FAIL");
    }
    printf("\n## Perf (%d-iter CUDA-event mean, block_size=32, batch=1)\n\n", config.iters);
    printf("| shape | heads | ctx | ref us | v2 us | speedup |\n");
    printf("|---|---|---|---|---|---|\n");
    for (const Row& r : rows) {
        if (!r.p.perf) continue;
        printf("| %s | %dq/%dkv | %d | %.2f | %.2f | %.2fx |\n",
               r.p.tag, r.p.H, r.p.KVH, r.p.ctx, r.ref_us, r.v2_us,
               r.ref_us / r.v2_us);
    }
    printf("\nRESULT: parity %s; timing reported without a promotion threshold\n",
           all_pass ? "ALL PASS" : "FAILURES");

    int flags = O_WRONLY | O_CREAT | O_EXCL;
#ifdef O_NOFOLLOW
    flags |= O_NOFOLLOW;
#endif
    int fd = open(config.json_path, flags, 0600);
    if (fd < 0) {
        fprintf(stderr, "cannot create JSON result %s: %s\n", config.json_path, strerror(errno));
        return 2;
    }
    FILE* jf = fdopen(fd, "w");
    if (jf == nullptr) {
        fprintf(stderr, "fdopen failed for %s: %s\n", config.json_path, strerror(errno));
        close(fd);
        return 2;
    }
    fprintf(jf, "{\"schema\":\"rvllm-fp8-decode-parity-v1\",\"seed\":%u,", config.seed);
    fprintf(jf, "\"tolerances\":{\"absolute\":%.17g,\"relative\":%.17g},",
            config.abs_tol, config.rel_tol);
    fprintf(jf, "\"warmup\":%d,\"iterations\":%d,", config.warmup, config.iters);
    fprintf(jf, "\"toolchain\":{\"cudart_compile\":%d,\"driver\":%d,\"runtime\":%d},",
            CUDART_VERSION, driver_version, runtime_version);
    fprintf(jf, "\"device\":{\"name\":");
    json_string(jf, prop.name);
    fprintf(jf, ",\"compute_capability\":\"%d.%d\"},", prop.major, prop.minor);
    fprintf(jf, "\"conversion_pass\":%s,\"rows\":[", conv_pass ? "true" : "false");
    for (size_t i = 0; i < rows.size(); i++) {
        const Row& r = rows[i];
        if (i) fputc(',', jf);
        fprintf(jf, "{\"shape\":"); json_string(jf, r.p.tag);
        fprintf(jf, ",\"context\":%d,\"block_size\":%d,\"batch\":%d,",
                r.p.ctx, r.p.bs, r.p.batch);
        fprintf(jf, "\"null_qscale\":%s,\"null_kvscale\":%s,",
                r.p.null_qscale ? "true" : "false", r.p.null_kvscale ? "true" : "false");
        fprintf(jf, "\"max_abs\":%.17g,\"max_rel\":%.17g,\"pass\":%s,",
                r.max_abs, r.max_rel, r.pass ? "true" : "false");
        fprintf(jf, "\"reference_us\":%.17g,\"candidate_us\":%.17g}", r.ref_us, r.v2_us);
    }
    fprintf(jf, "],\"pass\":%s}\n", all_pass ? "true" : "false");
    const int flush_rc = fflush(jf);
    const int sync_rc = flush_rc == 0 ? fsync(fd) : -1;
    const int close_rc = fclose(jf);
    if (flush_rc != 0 || sync_rc != 0 || close_rc != 0) {
        fprintf(stderr, "failed to finalize %s\n", config.json_path);
        return 2;
    }
    return all_pass ? 0 : 1;
}
