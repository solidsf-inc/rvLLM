#include <stdint.h>

extern "C" __global__ void map_i32_token_id_kernel(
    const int* __restrict__ row_id,
    int* __restrict__ token_id,
    const int* __restrict__ keep_ids,
    int keep_len
) {
    if (blockIdx.x != 0 || blockIdx.y != 0 || blockIdx.z != 0 ||
        threadIdx.x != 0 || threadIdx.y != 0 || threadIdx.z != 0) return;
    if (token_id == nullptr) return;
    if (row_id == nullptr || keep_ids == nullptr || keep_len <= 0) {
        token_id[0] = -1;
        return;
    }
    int row = row_id[0];
    token_id[0] = (row >= 0 && row < keep_len) ? keep_ids[row] : -1;
}
