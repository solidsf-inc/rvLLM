//! SIMD-accelerated math for LLM sampling.
//! Uses @Vector for portable SIMD: NEON on aarch64, AVX-512 on x86_64.

const std = @import("std");

// Vector width: 16 floats (512-bit on AVX-512, split to 4x128 on NEON).
const V = 16;
const Vec = @Vector(V, f32);
const IdxVec = @Vector(V, u32);

const neg_inf: f32 = -std.math.inf(f32);
const neg_inf_vec: Vec = @splat(neg_inf);

// -- softmax ------------------------------------------------------------------

/// 3-pass numerically stable softmax. Reads `src`, writes `dst`.
pub fn softmax(src: []const f32, dst: []f32) void {
    std.debug.assert(src.len == dst.len);
    const n = src.len;
    if (n == 0) return;

    if (firstNaN(src) != null) {
        @memset(dst, std.math.nan(f32));
        return;
    }

    // Pass 1: max
    const mx = maxVal(src);
    if (mx == std.math.inf(f32)) {
        var count: usize = 0;
        for (src) |value| count += @intFromBool(value == mx);
        const probability = 1.0 / @as(f32, @floatFromInt(count));
        for (src, dst) |value, *out| out.* = if (value == mx) probability else 0.0;
        return;
    }
    if (mx == neg_inf) {
        @memset(dst, std.math.nan(f32));
        return;
    }
    const mx_vec: Vec = @splat(mx);

    // Pass 2: exp(x - max) + accumulate sum
    var sum_vec: Vec = @splat(@as(f32, 0));
    var i: usize = 0;
    while (i + V <= n) : (i += V) {
        const v: Vec = src[i..][0..V].*;
        const e = @exp(v - mx_vec);
        dst[i..][0..V].* = e;
        sum_vec += e;
    }
    // Scalar tail
    var sum_tail: f32 = 0;
    while (i < n) : (i += 1) {
        const e = @exp(src[i] - mx);
        dst[i] = e;
        sum_tail += e;
    }
    const total = @reduce(.Add, sum_vec) + sum_tail;

    if (!(total > 0.0) or !std.math.isFinite(total)) {
        @memset(dst, std.math.nan(f32));
        return;
    }

    // Pass 3: normalize
    const inv: Vec = @splat(1.0 / total);
    i = 0;
    while (i + V <= n) : (i += V) {
        const v: Vec = dst[i..][0..V].*;
        dst[i..][0..V].* = v * inv;
    }
    const inv_s = 1.0 / total;
    while (i < n) : (i += 1) {
        dst[i] *= inv_s;
    }
}

// -- log_softmax --------------------------------------------------------------

/// Log-softmax: log(softmax(x)) = x - max - log(sum(exp(x - max)))
pub fn logSoftmax(src: []const f32, dst: []f32) void {
    std.debug.assert(src.len == dst.len);
    const n = src.len;
    if (n == 0) return;

    if (firstNaN(src) != null) {
        @memset(dst, std.math.nan(f32));
        return;
    }

    const mx = maxVal(src);
    if (mx == std.math.inf(f32)) {
        var count: usize = 0;
        for (src) |value| count += @intFromBool(value == mx);
        const selected = -@log(@as(f32, @floatFromInt(count)));
        for (src, dst) |value, *out| out.* = if (value == mx) selected else neg_inf;
        return;
    }
    if (mx == neg_inf) {
        @memset(dst, std.math.nan(f32));
        return;
    }
    const mx_vec: Vec = @splat(mx);

    // Accumulate sum(exp(x - max))
    var sum_vec: Vec = @splat(@as(f32, 0));
    var i: usize = 0;
    while (i + V <= n) : (i += V) {
        const v: Vec = src[i..][0..V].*;
        sum_vec += @exp(v - mx_vec);
    }
    var sum_tail: f32 = 0;
    while (i < n) : (i += 1) {
        sum_tail += @exp(src[i] - mx);
    }
    const total = @reduce(.Add, sum_vec) + sum_tail;
    if (!(total > 0.0) or !std.math.isFinite(total)) {
        @memset(dst, std.math.nan(f32));
        return;
    }
    const lse = mx + @log(total);
    const lse_vec: Vec = @splat(lse);

    // x - lse
    i = 0;
    while (i + V <= n) : (i += V) {
        const v: Vec = src[i..][0..V].*;
        dst[i..][0..V].* = v - lse_vec;
    }
    while (i < n) : (i += 1) {
        dst[i] = src[i] - lse;
    }
}

// -- argmax -------------------------------------------------------------------

/// Returns the first index of the maximum element. A NaN is treated as an
/// invalid logit and its first index is returned. Returns 0 for an empty slice.
pub fn argmax(data: []const f32) u32 {
    const n = data.len;
    if (n == 0) return 0;
    std.debug.assert(n <= std.math.maxInt(u32));
    if (firstNaN(data)) |index| return index;

    // SIMD path: track max values and their indices per lane
    var best_vals: Vec = @splat(neg_inf);
    var best_idxs: IdxVec = @splat(std.math.maxInt(u32));

    // Base index vector: [0, 1, 2, ..., V-1]
    var base_idx: IdxVec = undefined;
    inline for (0..V) |j| {
        base_idx[j] = @intCast(j);
    }
    const stride: IdxVec = @splat(V);

    var i: usize = 0;
    while (i + V <= n) : (i += V) {
        const v: Vec = data[i..][0..V].*;
        const mask = (v > best_vals) | ((v == best_vals) & (base_idx < best_idxs));
        best_vals = @select(f32, mask, v, best_vals);
        best_idxs = @select(u32, mask, base_idx, best_idxs);
        base_idx += stride;
    }

    // Reduce lanes
    var best_val: f32 = neg_inf;
    var best_idx: u32 = std.math.maxInt(u32);
    inline for (0..V) |j| {
        if (best_vals[j] > best_val or
            (best_vals[j] == best_val and best_idxs[j] < best_idx))
        {
            best_val = best_vals[j];
            best_idx = best_idxs[j];
        }
    }

    // Scalar tail
    while (i < n) : (i += 1) {
        const index: u32 = @intCast(i);
        if (data[i] > best_val or (data[i] == best_val and index < best_idx)) {
            best_val = data[i];
            best_idx = index;
        }
    }
    return if (best_idx == std.math.maxInt(u32)) 0 else best_idx;
}

// -- max ----------------------------------------------------------------------

/// Maximum value in slice. Returns -inf for empty.
pub fn maxVal(data: []const f32) f32 {
    const n = data.len;
    if (n == 0) return neg_inf;
    if (firstNaN(data) != null) return std.math.nan(f32);

    var acc: Vec = @splat(neg_inf);
    var i: usize = 0;
    while (i + V <= n) : (i += V) {
        const v: Vec = data[i..][0..V].*;
        acc = @max(acc, v);
    }
    var best = @reduce(.Max, acc);
    while (i < n) : (i += 1) {
        if (data[i] > best) best = data[i];
    }
    return best;
}

// -- argmax + logprob (fused) -------------------------------------------------

/// Fused argmax + log-probability in 2 passes (instead of 4 separate).
/// Pass 1: max + argmax. Pass 2: sum(exp(x - max)).
/// Returns (argmax_index, logprob_of_argmax).
pub fn argmaxLogprob(data: []const f32) struct { idx: u32, logprob: f32 } {
    const n = data.len;
    if (n == 0) return .{ .idx = 0, .logprob = 0.0 };
    std.debug.assert(n <= std.math.maxInt(u32));
    if (firstNaN(data)) |index| {
        return .{ .idx = index, .logprob = std.math.nan(f32) };
    }

    // Pass 1: simultaneous max + argmax
    var best_vals: Vec = @splat(neg_inf);
    var best_idxs: IdxVec = @splat(std.math.maxInt(u32));
    var base_idx: IdxVec = undefined;
    inline for (0..V) |j| {
        base_idx[j] = @intCast(j);
    }
    const stride: IdxVec = @splat(V);

    var i: usize = 0;
    while (i + V <= n) : (i += V) {
        const v: Vec = data[i..][0..V].*;
        const mask = (v > best_vals) | ((v == best_vals) & (base_idx < best_idxs));
        best_vals = @select(f32, mask, v, best_vals);
        best_idxs = @select(u32, mask, base_idx, best_idxs);
        base_idx += stride;
    }

    var best_val: f32 = neg_inf;
    var best_idx: u32 = std.math.maxInt(u32);
    inline for (0..V) |j| {
        if (best_vals[j] > best_val or
            (best_vals[j] == best_val and best_idxs[j] < best_idx))
        {
            best_val = best_vals[j];
            best_idx = best_idxs[j];
        }
    }
    while (i < n) : (i += 1) {
        const index: u32 = @intCast(i);
        if (data[i] > best_val or (data[i] == best_val and index < best_idx)) {
            best_val = data[i];
            best_idx = index;
        }
    }

    if (best_val == std.math.inf(f32)) {
        var count: usize = 0;
        for (data) |value| count += @intFromBool(value == best_val);
        return .{
            .idx = best_idx,
            .logprob = -@log(@as(f32, @floatFromInt(count))),
        };
    }
    if (best_val == neg_inf) {
        return .{ .idx = 0, .logprob = std.math.nan(f32) };
    }

    // Pass 2: sum(exp(x - max))
    const mx_vec: Vec = @splat(best_val);
    var sum_vec: Vec = @splat(@as(f32, 0));
    i = 0;
    while (i + V <= n) : (i += V) {
        const v: Vec = data[i..][0..V].*;
        sum_vec += @exp(v - mx_vec);
    }
    var sum_tail: f32 = 0;
    while (i < n) : (i += 1) {
        sum_tail += @exp(data[i] - best_val);
    }
    const total = @reduce(.Add, sum_vec) + sum_tail;

    // logprob = (x[idx] - max) - ln(sum) = 0 - ln(sum) = -ln(sum)
    const logprob = -@log(total);

    return .{ .idx = best_idx, .logprob = logprob };
}

fn firstNaN(data: []const f32) ?u32 {
    for (data, 0..) |value, index| {
        if (std.math.isNan(value)) return @intCast(index);
    }
    return null;
}

// -- scale --------------------------------------------------------------------

/// In-place multiply by scalar (temperature scaling: inv_temp = 1/T).
pub fn scale(data: []f32, factor: f32) void {
    const n = data.len;
    const fv: Vec = @splat(factor);
    var i: usize = 0;
    while (i + V <= n) : (i += V) {
        const v: Vec = data[i..][0..V].*;
        data[i..][0..V].* = v * fv;
    }
    while (i < n) : (i += 1) {
        data[i] *= factor;
    }
}

test "argmax tie uses first index across SIMD lanes" {
    var values = [_]f32{0.0} ** 32;
    values[15] = 5.0;
    values[16] = 5.0;
    try std.testing.expectEqual(@as(u32, 15), argmax(&values));
    try std.testing.expectEqual(@as(u32, 15), argmaxLogprob(&values).idx);
}

test "non-finite softmax behavior is explicit" {
    const all_masked = [_]f32{ neg_inf, neg_inf };
    var out: [2]f32 = undefined;
    softmax(&all_masked, &out);
    try std.testing.expect(std.math.isNan(out[0]));
    try std.testing.expect(std.math.isNan(out[1]));

    const positive_infinity = [_]f32{ std.math.inf(f32), 1.0, std.math.inf(f32) };
    var selected: [3]f32 = undefined;
    softmax(&positive_infinity, &selected);
    try std.testing.expectEqual(@as(f32, 0.5), selected[0]);
    try std.testing.expectEqual(@as(f32, 0.0), selected[1]);
    try std.testing.expectEqual(@as(f32, 0.5), selected[2]);

    const with_nan = [_]f32{ 1.0, std.math.nan(f32), 2.0 };
    softmax(&with_nan, &selected);
    try std.testing.expect(std.math.isNan(selected[0]));
    try std.testing.expectEqual(@as(u32, 1), argmax(&with_nan));
}
