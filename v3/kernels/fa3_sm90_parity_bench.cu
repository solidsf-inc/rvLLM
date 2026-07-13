// Standalone H100 conformance test for the rvLLM FA3 and fallback ABIs.
//
// Build:
//   nvcc -std=c++17 -O3 -lineinfo -arch=sm_90a \
//     v3/kernels/fa3_sm90_parity_bench.cu -ldl -o /tmp/fa3_sm90_parity_bench
// Run from the repository root:
//   /tmp/fa3_sm90_parity_bench

// Optional arguments override the default FA3 and fallback shared-library paths.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

static int failures = 0;

static void cuda_check(cudaError_t error, const char* expression, int line) {
    if (error == cudaSuccess) return;
    std::fprintf(stderr, "CUDA failure at line %d: %s: %s\n", line, expression,
                 cudaGetErrorString(error));
    std::exit(2);
}

#define CUDA_CHECK(expression) cuda_check((expression), #expression, __LINE__)

static void check(bool condition, const char* label) {
    std::printf("[%s] %s\n", condition ? "PASS" : "FAIL", label);
    failures += !condition;
}

struct DeviceBuffer {
    void* data = nullptr;
    size_t bytes = 0;

    explicit DeviceBuffer(size_t size = 0) : bytes(size) {
        if (bytes) CUDA_CHECK(cudaMalloc(&data, bytes));
    }

    ~DeviceBuffer() {
        if (data) cudaFree(data);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

class Library {
  public:
    Library(const char* path, const char* name) : name_(name) {
        handle_ = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!handle_) {
            std::fprintf(stderr, "cannot load %s (%s): %s\n", name, path, dlerror());
            std::exit(2);
        }
    }

    ~Library() { dlclose(handle_); }

    template <class T>
    T symbol(const char* symbol_name) {
        dlerror();
        void* address = dlsym(handle_, symbol_name);
        const char* error = dlerror();
        if (error || !address) {
            std::fprintf(stderr, "%s lacks %s: %s\n", name_, symbol_name,
                         error ? error : "null symbol");
            std::exit(2);
        }
        T function = nullptr;
        static_assert(sizeof(function) == sizeof(address), "function pointer size");
        std::memcpy(&function, &address, sizeof(function));
        return function;
    }

  private:
    void* handle_ = nullptr;
    const char* name_ = nullptr;
};

using AbiFn = int (*)();
using RevisionFn = const char* (*)();
using DtypeFn = int (*)();
using Fa3DecodeWorkspaceFn = uint64_t (*)(int, int, int, int, int, int, int, int);
using Fa3PrefillWorkspaceFn = uint64_t (*)(int, int, int, int, int, int, int, int,
                                           int, int);
using Sm89DecodeWorkspaceFn = uint64_t (*)(int, int, int, int);

using DecodeFn = int (*)(void*, void*, void*, void*, int*, int*, void*, size_t,
                         float, int, int, int, int, int, int, int, int,
                         cudaStream_t);
using DecodeFp8Fn = int (*)(void*, void*, void*, void*, int*, int*, void*, size_t,
                            void*, void*, void*, float*, float*, float*, float,
                            int, int, int, int, int, int, int, int, cudaStream_t);
using PrefillFp8Fn = int (*)(void*, void*, void*, void*, int*, int*, int*, void*,
                             size_t, void*, void*, void*, float*, float*, float*,
                             float, int, int, int, int, int, int, int, int, int,
                             int, cudaStream_t);
using Sm89DecodeFn = int (*)(void*, void*, void*, void*, void*, void*, void*, size_t,
                             float, int, int, int, int, int, int, int, int, void*);
using Sm89DecodeFp8Fn = int (*)(void*, void*, void*, void*, void*, void*, void*,
                                size_t, void*, void*, void*, float*, float*, float*,
                                float, int, int, int, int, int, int, int, int, void*);
using Sm89PrefillFp8Fn = int (*)(void*, void*, void*, void*, void*, void*, void*,
                                 void*, size_t, void*, void*, void*, float*, float*,
                                 float*, float, int, int, int, int, int, int, int,
                                 int, int, int, void*);

struct Fa3Api {
    AbiFn abi;
    RevisionFn revision;
    DtypeFn fp8_dtype;
    DtypeFn fp8_element_size;
    Fa3DecodeWorkspaceFn decode_workspace;
    Fa3PrefillWorkspaceFn prefill_workspace;
    DecodeFn decode;
    DecodeFp8Fn decode_fp8;
    PrefillFp8Fn prefill_fp8;
};

struct Sm89Api {
    AbiFn abi;
    DtypeFn fp8_dtype;
    DtypeFn fp8_element_size;
    Sm89DecodeWorkspaceFn decode_workspace;
    Sm89DecodeFn decode;
    Sm89DecodeFp8Fn decode_fp8;
    Sm89PrefillFp8Fn prefill_fp8;
};

static size_t align256(size_t value) { return (value + 255) & ~size_t(255); }

static size_t fa3_base_workspace(int rows, int batch, int heads, bool prefill,
                                 bool local) {
    const int rounded_batch = (batch + 3) / 4 * 4;
    const int metadata_vectors = 2 + !local + (prefill || local);
    return align256((size_t(rounded_batch) * metadata_vectors + 1) * sizeof(int)) +
           align256(size_t(rows) * heads * sizeof(float));
}

static float half_to_float(uint16_t h) {
    const int sign = h >> 15;
    const int exponent = (h >> 10) & 31;
    const int mantissa = h & 1023;
    float value;
    if (exponent == 31) {
        value = mantissa ? NAN : INFINITY;
    } else if (exponent == 0) {
        value = std::ldexp(float(mantissa), -24);
    } else {
        value = std::ldexp(float(1024 + mantissa), exponent - 25);
    }
    return sign ? -value : value;
}

static float fp8_to_float(uint8_t x) {
    const int sign = x >> 7;
    const int exponent = (x >> 3) & 15;
    const int mantissa = x & 7;
    if (exponent == 15 && mantissa == 7) return NAN;
    float value = exponent == 0
        ? std::ldexp(float(mantissa), -9)
        : std::ldexp(1.0f + float(mantissa) / 8.0f, exponent - 7);
    return sign ? -value : value;
}

static uint32_t mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

struct Inputs {
    std::vector<unsigned char> q_raw;
    std::vector<unsigned char> k_raw;
    std::vector<unsigned char> v_raw;
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
};

static Inputs make_inputs(size_t q_elements, size_t kv_elements, bool fp8,
                          uint32_t seed, float q_descale, float k_descale,
                          float v_descale) {
    Inputs result;
    result.q.resize(q_elements);
    result.k.resize(kv_elements);
    result.v.resize(kv_elements);

    if (fp8) {
        static const uint8_t values[] = {
            0x18, 0x20, 0x28, 0x30, 0x34, 0x38, 0x3c, 0x40,
            0x98, 0xa0, 0xa8, 0xb0, 0xb4, 0xb8, 0xbc, 0xc0,
        };
        result.q_raw.resize(q_elements);
        result.k_raw.resize(kv_elements);
        result.v_raw.resize(kv_elements);
        for (size_t i = 0; i < q_elements; ++i) {
            uint8_t x = values[mix32(uint32_t(i) + seed) & 15];
            result.q_raw[i] = x;
            result.q[i] = fp8_to_float(x) * q_descale;
        }
        for (size_t i = 0; i < kv_elements; ++i) {
            uint8_t k = values[mix32(uint32_t(i) + seed + 0x13579u) & 15];
            uint8_t v = values[mix32(uint32_t(i) + seed + 0x2468au) & 15] & 0x7f;
            result.k_raw[i] = k;
            result.v_raw[i] = v;
            result.k[i] = fp8_to_float(k) * k_descale;
            result.v[i] = fp8_to_float(v) * v_descale;
        }
        return result;
    }

    result.q_raw.resize(q_elements * sizeof(uint16_t));
    result.k_raw.resize(kv_elements * sizeof(uint16_t));
    result.v_raw.resize(kv_elements * sizeof(uint16_t));
    auto fill_half = [&](std::vector<unsigned char>& raw, std::vector<float>& decoded,
                         uint32_t salt, float magnitude) {
        for (size_t i = 0; i < decoded.size(); ++i) {
            int centered = int(mix32(uint32_t(i) + seed + salt) % 2001) - 1000;
            __half h = __float2half_rn(float(centered) * magnitude / 1000.0f);
            uint16_t bits;
            std::memcpy(&bits, &h, sizeof(bits));
            std::memcpy(raw.data() + i * sizeof(bits), &bits, sizeof(bits));
            decoded[i] = half_to_float(bits);
        }
    };
    fill_half(result.q_raw, result.q, 0x10203u, 0.25f);
    fill_half(result.k_raw, result.k, 0x40506u, 0.25f);
    fill_half(result.v_raw, result.v, 0x70809u, 0.75f);
    for (size_t i = 0; i < result.v.size(); ++i) {
        uint16_t bits;
        std::memcpy(&bits, result.v_raw.data() + i * sizeof(bits), sizeof(bits));
        bits &= 0x7fff;
        std::memcpy(result.v_raw.data() + i * sizeof(bits), &bits, sizeof(bits));
        result.v[i] = half_to_float(bits);
    }
    return result;
}

static std::vector<float> reference_attention(
    const Inputs& inputs, const std::vector<int>& block_tables,
    const std::vector<int>& context_lens, const std::vector<int>& cu_seqlens_q,
    int batch, int heads, int kv_heads, int head_dim, int block_size,
    int max_blocks_per_seq, int window_size_left, float scale) {
    const bool prefill = !cu_seqlens_q.empty();
    const int rows = prefill ? cu_seqlens_q.back() : batch;
    std::vector<float> output(size_t(rows) * heads * head_dim);

    for (int row = 0; row < rows; ++row) {
        int sequence = row;
        int position = 0;
        int q_length = 1;
        if (prefill) {
            sequence = int(std::upper_bound(cu_seqlens_q.begin(), cu_seqlens_q.end(),
                                            row) - cu_seqlens_q.begin()) - 1;
            position = row - cu_seqlens_q[sequence];
            q_length = cu_seqlens_q[sequence + 1] - cu_seqlens_q[sequence];
        }
        const int context = context_lens[sequence];
        int attend_end = prefill ? context - q_length + position + 1 : context;
        attend_end = std::min(attend_end, context);
        int attend_start = 0;
        if (window_size_left >= 0)
            attend_start = std::max(0, attend_end - window_size_left - 1);

        for (int head = 0; head < heads; ++head) {
            const int kv_head = head * kv_heads / heads;
            const size_t q_offset = (size_t(row) * heads + head) * head_dim;
            const int tokens = attend_end - attend_start;
            std::vector<double> logits(tokens);
            double maximum = -std::numeric_limits<double>::infinity();

            for (int token = attend_start; token < attend_end; ++token) {
                const int logical_page = token / block_size;
                const int table_page = window_size_left >= 0
                    ? logical_page % max_blocks_per_seq : logical_page;
                const int physical_page =
                    block_tables[size_t(sequence) * max_blocks_per_seq + table_page];
                const int offset_in_page = token % block_size;
                const size_t kv_offset =
                    ((size_t(physical_page) * block_size + offset_in_page) * kv_heads +
                     kv_head) * head_dim;
                double dot = 0.0;
                for (int d = 0; d < head_dim; ++d)
                    dot += double(inputs.q[q_offset + d]) * inputs.k[kv_offset + d];
                dot *= scale;
                logits[token - attend_start] = dot;
                maximum = std::max(maximum, dot);
            }

            std::vector<double> accumulator(head_dim, 0.0);
            double denominator = 0.0;
            for (int token = attend_start; token < attend_end; ++token) {
                const double weight = std::exp(logits[token - attend_start] - maximum);
                denominator += weight;
                const int logical_page = token / block_size;
                const int table_page = window_size_left >= 0
                    ? logical_page % max_blocks_per_seq : logical_page;
                const int physical_page =
                    block_tables[size_t(sequence) * max_blocks_per_seq + table_page];
                const int offset_in_page = token % block_size;
                const size_t kv_offset =
                    ((size_t(physical_page) * block_size + offset_in_page) * kv_heads +
                     kv_head) * head_dim;
                for (int d = 0; d < head_dim; ++d)
                    accumulator[d] += weight * inputs.v[kv_offset + d];
            }
            for (int d = 0; d < head_dim; ++d)
                output[q_offset + d] = float(accumulator[d] / denominator);
        }
    }
    return output;
}

struct Difference {
    double max_absolute = 0.0;
    double max_relative = 0.0;
    size_t nonfinite = 0;
    size_t outside = 0;
};

static Difference compare_reference(const std::vector<uint16_t>& actual,
                                    const std::vector<float>& expected,
                                    double absolute_tolerance,
                                    double relative_tolerance) {
    Difference difference;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double got = half_to_float(actual[i]);
        const double want = expected[i];
        if (!std::isfinite(got) || !std::isfinite(want)) {
            ++difference.nonfinite;
            continue;
        }
        const double absolute = std::abs(got - want);
        const double relative = absolute / std::max(std::abs(want), 1e-6);
        difference.max_absolute = std::max(difference.max_absolute, absolute);
        difference.max_relative = std::max(difference.max_relative, relative);
        difference.outside += absolute > absolute_tolerance +
                                          relative_tolerance * std::abs(want);
    }
    return difference;
}

static Difference compare_outputs(const std::vector<uint16_t>& lhs,
                                  const std::vector<uint16_t>& rhs,
                                  double absolute_tolerance,
                                  double relative_tolerance) {
    std::vector<float> expected(rhs.size());
    for (size_t i = 0; i < rhs.size(); ++i) expected[i] = half_to_float(rhs[i]);
    return compare_reference(lhs, expected, absolute_tolerance, relative_tolerance);
}

static bool difference_passes(const Difference& difference) {
    return difference.nonfinite == 0 && difference.outside == 0;
}

static void print_difference(const char* label, const Difference& difference) {
    std::printf("        %s max_abs=%.4e max_rel=%.4e nonfinite=%zu outside=%zu\n",
                label, difference.max_absolute, difference.max_relative,
                difference.nonfinite, difference.outside);
}

struct DecodeShape {
    int max_blocks = 0;
    int context = 0;
    int window = -1;
    size_t workspace = 0;
};

struct PrefillShape {
    int max_blocks = 0;
    int window = -1;
    size_t workspace = 0;
};

static DecodeShape pick_decode_shape(const Fa3Api& fa3, int head_dim, bool fp8,
                                     bool local, bool want_split) {
    constexpr int batch = 1;
    constexpr int heads = 4;
    constexpr int kv_heads = 1;
    constexpr int block_size = 32;
    static const int candidates[] = {
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
    };
    const size_t base = fa3_base_workspace(batch, batch, heads, false, local);
    for (int blocks : candidates) {
        const int capacity = blocks * block_size;
        const int window = local ? std::max(1, capacity / 2 - 1) : -1;
        uint64_t queried = fa3.decode_workspace(
            batch, heads, kv_heads, head_dim, block_size, blocks, fp8, window);
        if (!queried || queried > SIZE_MAX) continue;
        if ((queried > base) == want_split)
            return {blocks, std::max(1, capacity - 3), window, size_t(queried)};
    }
    return {};
}

static PrefillShape pick_prefill_shape(const Fa3Api& fa3, int head_dim,
                                       bool local, bool want_split) {
    constexpr int total_q = 8;
    constexpr int max_q = 5;
    constexpr int batch = 2;
    constexpr int heads = 4;
    constexpr int kv_heads = 1;
    constexpr int block_size = 32;
    static const int candidates[] = {
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
    };
    const size_t base = fa3_base_workspace(total_q, batch, heads, true, local);
    for (int blocks : candidates) {
        const int capacity = blocks * block_size;
        const int window = local
            ? (want_split ? std::max(1, capacity / 2 - 1) : 3)
            : -1;
        uint64_t queried = fa3.prefill_workspace(
            total_q, max_q, batch, heads, kv_heads, head_dim, block_size,
            blocks, 1, window);
        if (!queried || queried > SIZE_MAX) continue;
        if ((queried > base) == want_split)
            return {blocks, window, size_t(queried)};
    }
    return {};
}

static std::vector<uint16_t> copy_output(DeviceBuffer& output, size_t elements,
                                         cudaStream_t stream) {
    std::vector<uint16_t> result(elements);
    CUDA_CHECK(cudaMemcpyAsync(result.data(), output.data, elements * sizeof(uint16_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return result;
}

static void run_decode_case(const Fa3Api& fa3, const Sm89Api& sm89, int head_dim,
                            bool fp8, bool local, bool want_split,
                            cudaStream_t stream, uint32_t seed) {
    constexpr int batch = 1;
    constexpr int heads = 4;
    constexpr int kv_heads = 1;
    constexpr int block_size = 32;
    const DecodeShape shape =
        pick_decode_shape(fa3, head_dim, fp8, local, want_split);
    const char* dtype = fp8 ? "FP8" : "F16";
    const char* attention = local ? "sliding" : "global";
    const char* split = want_split ? "split" : "non-split";
    char label[160];
    std::snprintf(label, sizeof(label), "decode HD%d %s %s %s", head_dim, dtype,
                  attention, split);
    if (!shape.max_blocks) {
        check(false, label);
        return;
    }

    const size_t cache_elements =
        size_t(shape.max_blocks) * block_size * kv_heads * head_dim;
    const size_t q_elements = size_t(batch) * heads * head_dim;
    const size_t output_elements = q_elements;
    constexpr float q_descale = 0.25f;
    constexpr float k_descale = 0.25f;
    constexpr float v_descale = 0.50f;
    const double absolute_tolerance = fp8 ? 0.015 : 0.010;
    const double relative_tolerance = fp8 ? 0.050 : 0.035;
    const float softmax_scale = 1.0f / std::sqrt(float(head_dim));
    Inputs inputs = make_inputs(q_elements, cache_elements, fp8, seed,
                                q_descale, k_descale, v_descale);
    std::vector<int> block_tables(size_t(batch) * shape.max_blocks);
    for (int i = 0; i < shape.max_blocks; ++i) block_tables[i] = i;
    std::vector<int> context_lens(batch, shape.context);
    std::vector<int> no_cu_seqlens;
    std::vector<float> reference = reference_attention(
        inputs, block_tables, context_lens, no_cu_seqlens, batch, heads, kv_heads,
        head_dim, block_size, shape.max_blocks, shape.window, softmax_scale);
    const float reference_peak = *std::max_element(
        reference.begin(), reference.end(),
        [](float lhs, float rhs) { return std::abs(lhs) < std::abs(rhs); });
    check(std::abs(reference_peak) > 4.0 * absolute_tolerance,
          (std::string(label) + " reference signal").c_str());

    DeviceBuffer d_q(inputs.q_raw.size());
    DeviceBuffer d_k(inputs.k_raw.size());
    DeviceBuffer d_v(inputs.v_raw.size());
    DeviceBuffer d_bt(block_tables.size() * sizeof(int));
    DeviceBuffer d_cl(context_lens.size() * sizeof(int));
    DeviceBuffer d_fa_workspace(shape.workspace);
    DeviceBuffer d_scales(3 * sizeof(float));
    DeviceBuffer d_fa_a(output_elements * sizeof(uint16_t));
    DeviceBuffer d_fa_b(output_elements * sizeof(uint16_t));
    DeviceBuffer d_sm_a(output_elements * sizeof(uint16_t));
    DeviceBuffer d_sm_b(output_elements * sizeof(uint16_t));
    CUDA_CHECK(cudaMemcpy(d_q.data, inputs.q_raw.data(), inputs.q_raw.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k.data, inputs.k_raw.data(), inputs.k_raw.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v.data, inputs.v_raw.data(), inputs.v_raw.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bt.data, block_tables.data(), d_bt.bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cl.data, context_lens.data(), d_cl.bytes,
                          cudaMemcpyHostToDevice));
    const float scales[3] = {q_descale, k_descale, v_descale};
    CUDA_CHECK(cudaMemcpy(d_scales.data, scales, sizeof(scales), cudaMemcpyHostToDevice));
    float* d_q_descale = static_cast<float*>(d_scales.data);
    float* d_k_descale = d_q_descale + 1;
    float* d_v_descale = d_q_descale + 2;

    auto run_fa3 = [&](DeviceBuffer& output, size_t workspace_bytes) {
        if (fp8) {
            return fa3.decode_fp8(
                d_q.data, d_k.data, d_v.data, output.data,
                static_cast<int*>(d_bt.data), static_cast<int*>(d_cl.data),
                d_fa_workspace.data, workspace_bytes, nullptr, nullptr, nullptr,
                d_q_descale, d_k_descale, d_v_descale, softmax_scale, batch,
                heads, kv_heads, head_dim, block_size, shape.max_blocks,
                shape.max_blocks, shape.window, stream);
        }
        return fa3.decode(
            d_q.data, d_k.data, d_v.data, output.data,
            static_cast<int*>(d_bt.data), static_cast<int*>(d_cl.data),
            d_fa_workspace.data, workspace_bytes, softmax_scale, batch, heads,
            kv_heads, head_dim, block_size, shape.max_blocks, shape.max_blocks,
            shape.window, stream);
    };

    const int fa_undersized = run_fa3(d_fa_a, shape.workspace - 1);
    check(fa_undersized == -11, (std::string(label) + " FA3 rejects undersized workspace").c_str());

    uint64_t sm_workspace_query = fp8
        ? sm89.decode_workspace(batch, heads, kv_heads, head_dim) : 0;
    if (sm_workspace_query == UINT64_MAX || sm_workspace_query > SIZE_MAX) {
        check(false, (std::string(label) + " fallback workspace query").c_str());
        return;
    }
    DeviceBuffer d_sm_workspace{size_t(sm_workspace_query)};
    auto run_sm89 = [&](DeviceBuffer& output, size_t workspace_bytes) {
        if (fp8) {
            return sm89.decode_fp8(
                d_q.data, d_k.data, d_v.data, output.data,
                static_cast<int*>(d_bt.data), static_cast<int*>(d_cl.data),
                d_sm_workspace.data, workspace_bytes, nullptr, nullptr, nullptr,
                d_q_descale, d_k_descale, d_v_descale, softmax_scale, batch,
                heads, kv_heads, head_dim, block_size, shape.max_blocks,
                shape.max_blocks, shape.window, stream);
        }
        return sm89.decode(
            d_q.data, d_k.data, d_v.data, output.data,
            static_cast<int*>(d_bt.data), static_cast<int*>(d_cl.data), nullptr, 0,
            softmax_scale, batch, heads, kv_heads, head_dim, block_size,
            shape.max_blocks, shape.max_blocks, shape.window, stream);
    };
    if (fp8 && sm_workspace_query) {
        int rc = run_sm89(d_sm_a, size_t(sm_workspace_query) - 1);
        check(rc != 0, (std::string(label) + " fallback rejects undersized workspace").c_str());
    }

    CUDA_CHECK(cudaMemsetAsync(d_fa_a.data, 0xa5, d_fa_a.bytes, stream));
    int fa_a_rc = run_fa3(d_fa_a, shape.workspace);
    CUDA_CHECK(cudaMemsetAsync(d_fa_b.data, 0x5a, d_fa_b.bytes, stream));
    int fa_b_rc = run_fa3(d_fa_b, shape.workspace);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());
    check(fa_a_rc == 0 && fa_b_rc == 0,
          (std::string(label) + " FA3 launch contract").c_str());

    CUDA_CHECK(cudaMemsetAsync(d_sm_a.data, 0xa5, d_sm_a.bytes, stream));
    int sm_a_rc = run_sm89(d_sm_a, size_t(sm_workspace_query));
    CUDA_CHECK(cudaMemsetAsync(d_sm_b.data, 0x5a, d_sm_b.bytes, stream));
    int sm_b_rc = run_sm89(d_sm_b, size_t(sm_workspace_query));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());
    check(sm_a_rc == 0 && sm_b_rc == 0,
          (std::string(label) + " fallback launch contract").c_str());

    std::vector<uint16_t> fa_a = copy_output(d_fa_a, output_elements, stream);
    std::vector<uint16_t> fa_b = copy_output(d_fa_b, output_elements, stream);
    std::vector<uint16_t> sm_a = copy_output(d_sm_a, output_elements, stream);
    std::vector<uint16_t> sm_b = copy_output(d_sm_b, output_elements, stream);
    check(fa_a == fa_b, (std::string(label) + " FA3 deterministic").c_str());
    check(sm_a == sm_b, (std::string(label) + " fallback deterministic").c_str());

    Difference fa_reference = compare_reference(
        fa_a, reference, absolute_tolerance, relative_tolerance);
    Difference sm_reference = compare_reference(
        sm_a, reference, absolute_tolerance, relative_tolerance);
    Difference fa_sm = compare_outputs(
        fa_a, sm_a, absolute_tolerance, relative_tolerance);
    print_difference("FA3/reference", fa_reference);
    print_difference("fallback/reference", sm_reference);
    print_difference("FA3/fallback", fa_sm);
    const bool parity = difference_passes(fa_reference) &&
                        difference_passes(sm_reference) && difference_passes(fa_sm);
    std::snprintf(label, sizeof(label),
                  "decode HD%d %s %s %s parity (ctx=%d window=%d blocks=%d ws=%zu)",
                  head_dim, dtype, attention, split, shape.context, shape.window,
                  shape.max_blocks, shape.workspace);
    check(parity, label);
}

static void run_prefill_case(const Fa3Api& fa3, const Sm89Api& sm89, int head_dim,
                             bool local, bool want_split, cudaStream_t stream,
                             uint32_t seed) {
    constexpr int batch = 2;
    constexpr int heads = 4;
    constexpr int kv_heads = 1;
    constexpr int block_size = 32;
    const PrefillShape shape =
        pick_prefill_shape(fa3, head_dim, local, want_split);
    const char* attention = local ? "sliding" : "global";
    const char* split = want_split ? "split" : "non-split";
    char label[192];
    std::snprintf(label, sizeof(label), "prefill HD%d FP8 %s %s", head_dim,
                  attention, split);
    if (!shape.max_blocks) {
        check(false, label);
        return;
    }

    const int max_blocks = shape.max_blocks;
    const int num_blocks_total = batch * max_blocks;
    const int capacity = max_blocks * block_size;
    const int window = shape.window;
    const std::vector<int> cu_seqlens = {0, 3, 8};
    const std::vector<int> context_lens = {capacity - 7, capacity - 3};
    const int total_q = cu_seqlens.back();
    const int max_q = 5;
    const size_t q_elements = size_t(total_q) * heads * head_dim;
    const size_t cache_elements =
        size_t(num_blocks_total) * block_size * kv_heads * head_dim;
    const size_t output_elements = q_elements;
    constexpr float q_descale = 0.25f;
    constexpr float k_descale = 0.25f;
    constexpr float v_descale = 0.50f;
    constexpr double absolute_tolerance = 0.015;
    constexpr double relative_tolerance = 0.050;
    const float softmax_scale = 1.0f / std::sqrt(float(head_dim));
    Inputs inputs = make_inputs(q_elements, cache_elements, true, seed,
                                q_descale, k_descale, v_descale);
    std::vector<int> block_tables(size_t(batch) * max_blocks);
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < max_blocks; ++i)
            block_tables[size_t(b) * max_blocks + i] = b * max_blocks + i;
    std::vector<float> reference = reference_attention(
        inputs, block_tables, context_lens, cu_seqlens, batch, heads, kv_heads,
        head_dim, block_size, max_blocks, window, softmax_scale);

    const float reference_peak = *std::max_element(
        reference.begin(), reference.end(),
        [](float lhs, float rhs) { return std::abs(lhs) < std::abs(rhs); });
    check(std::abs(reference_peak) > 4.0 * absolute_tolerance,
          (std::string(label) + " reference signal").c_str());

    DeviceBuffer d_q(inputs.q_raw.size());
    DeviceBuffer d_k(inputs.k_raw.size());
    DeviceBuffer d_v(inputs.v_raw.size());
    DeviceBuffer d_bt(block_tables.size() * sizeof(int));
    DeviceBuffer d_cl(context_lens.size() * sizeof(int));
    DeviceBuffer d_cu(cu_seqlens.size() * sizeof(int));
    DeviceBuffer d_workspace{shape.workspace};
    DeviceBuffer d_scales(3 * sizeof(float));
    DeviceBuffer d_fa_a(output_elements * sizeof(uint16_t));
    DeviceBuffer d_fa_b(output_elements * sizeof(uint16_t));
    DeviceBuffer d_sm_a(output_elements * sizeof(uint16_t));
    DeviceBuffer d_sm_b(output_elements * sizeof(uint16_t));
    CUDA_CHECK(cudaMemcpy(d_q.data, inputs.q_raw.data(), inputs.q_raw.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k.data, inputs.k_raw.data(), inputs.k_raw.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v.data, inputs.v_raw.data(), inputs.v_raw.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bt.data, block_tables.data(), d_bt.bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cl.data, context_lens.data(), d_cl.bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cu.data, cu_seqlens.data(), d_cu.bytes, cudaMemcpyHostToDevice));
    const float scales[3] = {q_descale, k_descale, v_descale};
    CUDA_CHECK(cudaMemcpy(d_scales.data, scales, sizeof(scales), cudaMemcpyHostToDevice));
    float* d_q_descale = static_cast<float*>(d_scales.data);
    float* d_k_descale = d_q_descale + 1;
    float* d_v_descale = d_q_descale + 2;

    auto run_fa3 = [&](DeviceBuffer& output, size_t workspace_bytes) {
        return fa3.prefill_fp8(
            d_q.data, d_k.data, d_v.data, output.data,
            static_cast<int*>(d_bt.data), static_cast<int*>(d_cl.data),
            static_cast<int*>(d_cu.data), d_workspace.data, workspace_bytes,
            nullptr, nullptr, nullptr, d_q_descale, d_k_descale, d_v_descale,
            softmax_scale, total_q, max_q, batch, heads, kv_heads, head_dim,
            block_size, max_blocks, num_blocks_total, window, stream);
    };
    auto run_sm89 = [&](DeviceBuffer& output) {
        return sm89.prefill_fp8(
            d_q.data, d_k.data, d_v.data, output.data,
            static_cast<int*>(d_bt.data), static_cast<int*>(d_cl.data),
            static_cast<int*>(d_cu.data), nullptr, 0, nullptr, nullptr, nullptr,
            d_q_descale, d_k_descale, d_v_descale, softmax_scale, total_q, max_q,
            batch, heads, kv_heads, head_dim, block_size, max_blocks,
            num_blocks_total, window, stream);
    };

    check(run_fa3(d_fa_a, shape.workspace - 1) == -11,
          (std::string(label) + " FA3 rejects undersized workspace").c_str());
    CUDA_CHECK(cudaMemsetAsync(d_fa_a.data, 0xa5, d_fa_a.bytes, stream));
    int fa_a_rc = run_fa3(d_fa_a, shape.workspace);
    CUDA_CHECK(cudaMemsetAsync(d_fa_b.data, 0x5a, d_fa_b.bytes, stream));
    int fa_b_rc = run_fa3(d_fa_b, shape.workspace);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());
    check(fa_a_rc == 0 && fa_b_rc == 0,
          (std::string(label) + " FA3 launch contract").c_str());

    CUDA_CHECK(cudaMemsetAsync(d_sm_a.data, 0xa5, d_sm_a.bytes, stream));
    int sm_a_rc = run_sm89(d_sm_a);
    CUDA_CHECK(cudaMemsetAsync(d_sm_b.data, 0x5a, d_sm_b.bytes, stream));
    int sm_b_rc = run_sm89(d_sm_b);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());
    check(sm_a_rc == 0 && sm_b_rc == 0,
          (std::string(label) + " fallback launch contract").c_str());

    std::vector<uint16_t> fa_a = copy_output(d_fa_a, output_elements, stream);
    std::vector<uint16_t> fa_b = copy_output(d_fa_b, output_elements, stream);
    std::vector<uint16_t> sm_a = copy_output(d_sm_a, output_elements, stream);
    std::vector<uint16_t> sm_b = copy_output(d_sm_b, output_elements, stream);
    check(fa_a == fa_b, (std::string(label) + " FA3 deterministic").c_str());
    check(sm_a == sm_b, (std::string(label) + " fallback deterministic").c_str());

    Difference fa_reference = compare_reference(
        fa_a, reference, absolute_tolerance, relative_tolerance);
    Difference sm_reference = compare_reference(
        sm_a, reference, absolute_tolerance, relative_tolerance);
    Difference fa_sm = compare_outputs(
        fa_a, sm_a, absolute_tolerance, relative_tolerance);
    print_difference("FA3/reference", fa_reference);
    print_difference("fallback/reference", sm_reference);
    print_difference("FA3/fallback", fa_sm);
    std::snprintf(label, sizeof(label),
                  "prefill HD%d FP8 %s %s (ctx=%d/%d window=%d blocks=%d ws=%zu)",
                  head_dim, attention, split, context_lens[0], context_lens[1],
                  window, max_blocks, shape.workspace);
    check(difference_passes(fa_reference) && difference_passes(sm_reference) &&
              difference_passes(fa_sm),
          (std::string(label) + " parity").c_str());
}

int main(int argc, char** argv) {
    if (argc != 1 && argc != 3) {
        std::fprintf(stderr, "usage: %s [FA3_SO FALLBACK_SO]\n", argv[0]);
        return 2;
    }
    const char* fa3_path = argc == 3 ? argv[1] : "kernels/sm_90/libfa3_kernels.so";
    const char* sm89_path = argc == 3 ? argv[2] : "kernels/sm_90/libfa_sm89_kernels.so";

    cudaDeviceProp properties = {};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    if (properties.major != 9 || properties.minor != 0) {
        std::fprintf(stderr, "H100 (SM90) required; found %s SM%d%d\n", properties.name,
                     properties.major, properties.minor);
        return 2;
    }
    std::printf("device: %s SM%d%d\n", properties.name, properties.major, properties.minor);

    Library fa3_library(fa3_path, "FA3");
    Library sm89_library(sm89_path, "fallback");
    Fa3Api fa3 = {
        fa3_library.symbol<AbiFn>("rvllm_fa3_abi_version"),
        fa3_library.symbol<RevisionFn>("rvllm_fa3_upstream_revision"),
        fa3_library.symbol<DtypeFn>("fa3_sm90_fp8_output_dtype"),
        fa3_library.symbol<DtypeFn>("fa3_sm90_fp8_output_element_size"),
        fa3_library.symbol<Fa3DecodeWorkspaceFn>("fa3_sm90_decode_workspace_size"),
        fa3_library.symbol<Fa3PrefillWorkspaceFn>("fa3_sm90_prefill_workspace_size"),
        fa3_library.symbol<DecodeFn>("fa3_sm90_paged_decode"),
        fa3_library.symbol<DecodeFp8Fn>("fa3_sm90_paged_decode_fp8"),
        fa3_library.symbol<PrefillFp8Fn>("fa3_sm90_paged_prefill_fp8"),
    };
    Sm89Api sm89 = {
        sm89_library.symbol<AbiFn>("rvllm_fa_sm89_abi_version"),
        sm89_library.symbol<DtypeFn>("fa_sm89_fp8_output_dtype"),
        sm89_library.symbol<DtypeFn>("fa_sm89_fp8_output_element_size"),
        sm89_library.symbol<Sm89DecodeWorkspaceFn>("fa_sm89_decode_workspace_size"),
        sm89_library.symbol<Sm89DecodeFn>("fa_sm89_paged_decode"),
        sm89_library.symbol<Sm89DecodeFp8Fn>("fa_sm89_paged_decode_fp8"),
        sm89_library.symbol<Sm89PrefillFp8Fn>("fa_sm89_paged_prefill_fp8"),
    };

    const int fa3_abi = fa3.abi();
    const int sm89_abi = sm89.abi();
    check(fa3_abi == 2 && sm89_abi == 2, "ABI version 2");
    if (fa3_abi != 2 || sm89_abi != 2) {
        std::printf("RESULT: FAIL (%d failures)\n", failures);
        return 1;
    }
    const bool dtype_ok = fa3.fp8_dtype() == 1 && sm89.fp8_dtype() == 1;
    const bool element_size_ok =
        fa3.fp8_element_size() == 2 && sm89.fp8_element_size() == 2;
    check(dtype_ok, "FP8 output dtype is F16");
    check(element_size_ok, "FP8 output element size is two bytes");
    const char* revision = fa3.revision();
    const bool revision_ok = revision && std::strcmp(
        revision, "1233b73b6c95340c65c9edfe929611838354fc6e") == 0;
    check(revision_ok, "pinned FA3 upstream revision");
    if (!dtype_ok || !element_size_ok || !revision_ok) {
        std::printf("RESULT: FAIL (%d failures)\n", failures);
        return 1;
    }
    check(fa3.decode_workspace(0, 4, 1, 128, 32, 1, 0, -1) == 0,
          "FA3 invalid decode workspace query rejected");
    check(fa3.decode_workspace(1, 4, 1, 128, 32, 1, 0, 32) == 0,
          "FA3 sliding window beyond page capacity rejected");
    check(fa3.prefill_workspace(0, 1, 1, 4, 1, 128, 32, 1, 1, -1) == 0,
          "FA3 invalid prefill workspace query rejected");
    check(sm89.decode_workspace(0, 4, 1, 128) == UINT64_MAX,
          "fallback invalid workspace query rejected");

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    uint32_t seed = 0x5eed1234u;
    for (int head_dim : {128, 256}) {
        for (bool fp8 : {false, true}) {
            for (bool local : {false, true}) {
                for (bool split : {false, true}) {
                    run_decode_case(fa3, sm89, head_dim, fp8, local, split,
                                    stream, seed++);
                }
            }
        }
        run_prefill_case(fa3, sm89, head_dim, false, false, stream, seed++);
        run_prefill_case(fa3, sm89, head_dim, true, false, stream, seed++);
        if (head_dim == 128) {
            run_prefill_case(fa3, sm89, head_dim, false, true, stream, seed++);
            run_prefill_case(fa3, sm89, head_dim, true, true, stream, seed++);
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));

    std::printf("RESULT: %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
