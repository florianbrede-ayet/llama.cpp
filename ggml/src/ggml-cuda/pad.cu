#include "pad.cuh"

#include <stdint.h>

__device__ __forceinline__ int64_t wrap_around(int64_t coord, int64_t size) {
    // + size ensures negatives are handled properly
    return (coord + size) % size;
}

static __global__ void pad_f32(const float * src, size_t s00, size_t s01, size_t s02, size_t s03, float * dst,
                               const int lp0, const int rp0, const int lp1, const int rp1,
                               const int lp2, const int rp2, const int lp3, const int rp3,
                               const int ne0, const int ne1, const int ne2, const int ne3,
                               const bool circular) {
    // blockIdx.z: i3*ne2+i2
    // blockIdx.y: i1
    // blockIDx.x: i0 / CUDA_PAD_BLOCK_SIZE
    // gridDim.y:  ne1
    int i0 = threadIdx.x + blockIdx.x * blockDim.x;
    int i1 = blockIdx.y;
    int i2 = blockIdx.z % ne2;
    int i3 = blockIdx.z / ne2;

    if (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    const int64_t dst_idx = i3 * (ne0 * ne1 * ne2) + i2 * (ne0 * ne1) + i1 * ne0 + i0;

    if (!circular) {
        if ((i0 >= lp0 && i0 < ne0 - rp0) && (i1 >= lp1 && i1 < ne1 - rp1) && (i2 >= lp2 && i2 < ne2 - rp2) &&
            (i3 >= lp3 && i3 < ne3 - rp3)) {
            const int64_t i00  = i0 - lp0;
            const int64_t i01  = i1 - lp1;
            const int64_t i02  = i2 - lp2;
            const int64_t i03  = i3 - lp3;

            const int64_t src_idx = i03 * s03 + i02 * s02 + i01 * s01 + i00 * s00;

            dst[dst_idx] = src[src_idx];
        } else {
            dst[dst_idx] = 0.0f;
        }
    }
    // circular means on a torus, so x and y wrap around
    else {
        const int64_t ne00 = ne0 - lp0 - rp0;
        const int64_t ne01 = ne1 - lp1 - rp1;
        const int64_t ne02 = ne2 - lp2 - rp2;
        const int64_t ne03 = ne3 - lp3 - rp3;

        const int64_t i00 = wrap_around(i0 - lp0, ne00);
        const int64_t i01 = wrap_around(i1 - lp1, ne01);
        const int64_t i02 = wrap_around(i2 - lp2, ne02);
        const int64_t i03 = wrap_around(i3 - lp3, ne03);

        const int64_t src_idx = i03 * s03 + i02 * s02 + i01 * s01 + i00 * s00;

        dst[dst_idx] = src[src_idx];
    }
}


// optimized kernel for (0,2,1,3) permuted tensors where dim0 and dim2 are contiguous
static __global__ void pad_f32_permute021(
        const float * __restrict__ src, float * __restrict__ dst,
        const int ne0, const int ne1, const int ne2,
        const size_t s01, const size_t s03,
        const int ne0_padded, const int ne1_padded,
        const int rp0, const int rp1) {
    const int i1 = blockIdx.x;
    const int i3 = blockIdx.y;

    const float * src_base = src + (size_t)i3 * s03 + (size_t)i1 * s01;
    float * dst_base = dst + (size_t)i3 * ne2 * ne1_padded * ne0_padded;

    const int ne0_shift = __popc(ne0 - 1); // log2(ne0) [power of 2]
    const size_t dst_plane = (size_t)ne1_padded * ne0_padded;

    if (i1 < ne1) {
        const int total = ne0 * ne2;

        if (ne0 >= 4) {
            // float4 vectorized path for ne0 >= 4
            const int total4 = total >> 2;
            const float4 * src4 = (const float4 *)src_base;

            for (int idx4 = threadIdx.x; idx4 < total4; idx4 += blockDim.x) {
                const int idx = idx4 << 2;
                const int i0 = idx & (ne0 - 1);
                const int i2 = idx >> ne0_shift;

                const float4 val = src4[idx4];
                *((float4 *)(dst_base + i2 * dst_plane + (size_t)i1 * ne0_padded + i0)) = val;
            }
        } else {
            // scalar path for ne0 < 4
            for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
                const int i0 = idx & (ne0 - 1);
                const int i2 = idx >> ne0_shift;

                dst_base[i2 * dst_plane + (size_t)i1 * ne0_padded + i0] = src_base[idx];
            }
        }
    }

    // zerofill right padding for dim0 if needed
    if (rp0 > 0) {
        for (int i2 = 0; i2 < ne2; i2++) {
            float * row = dst_base + i2 * dst_plane + (size_t)i1 * ne0_padded;
            for (int i0 = ne0 + threadIdx.x; i0 < ne0_padded; i0 += blockDim.x) {
                row[i0] = 0.0f;
            }
        }
    }

    // zerofill padding rows for dim1
    if (rp1 > 0 && i1 >= ne1) {
        for (int i2 = 0; i2 < ne2; i2++) {
            float * row = dst_base + i2 * dst_plane + (size_t)i1 * ne0_padded;
            for (int i0 = threadIdx.x; i0 < ne0_padded; i0 += blockDim.x) {
                row[i0] = 0.0f;
            }
        }
    }
}

static void pad_f32_cuda(const float * src, size_t s00, size_t s01, size_t s02, size_t s03, float * dst,
    const int lp0, const int rp0, const int lp1, const int rp1,
    const int lp2, const int rp2, const int lp3, const int rp3,
    const int ne0, const int ne1, const int ne2, const int ne3,
    const bool circular, cudaStream_t stream) {

    // src dims
    const int ne00 = ne0 - lp0 - rp0;
    const int ne01 = ne1 - lp1 - rp1;
    const int ne02 = ne2 - lp2 - rp2;

    // fast path for (0,2,1,3) permuted tensors with conditions (satisfied by current qwen35 and qwen3next models):
    // circular [optional, mod ops], no left padding in dim0 [optional], no left padding in dim1 [optional], no dim2 padding,
    // no dim3 padding [optional], dim0 adjacent, ne00 non-empty, ne00 power of 2 [optional, div ops], s02 & s01 layout requirements
    if (!circular && lp0 == 0 && lp1 == 0 && lp2 == 0 && rp2 == 0 && lp3 == 0 && rp3 == 0 &&
        s00 == 1 && ne00 > 0 && (ne00 & (ne00 - 1)) == 0 &&
        s02 == (size_t)ne00 && s01 == (size_t)ne00 * ne02) {

        const int block_size = 256;
        dim3 gridDim(ne1, ne3, 1); // one block per (i1, i3), including padding rows
        pad_f32_permute021<<<gridDim, block_size, 0, stream>>>(
            src, dst, ne00, ne01, ne02, s01, s03, ne0, ne1, rp0, rp1);
        return;
    }

    int  num_blocks = (ne0 + CUDA_PAD_BLOCK_SIZE - 1) / CUDA_PAD_BLOCK_SIZE;
    dim3 gridDim(num_blocks, ne1, ne2 * ne3);
    pad_f32<<<gridDim, CUDA_PAD_BLOCK_SIZE, 0, stream>>>(src, s00, s01, s02, s03, dst,
                                                         lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3,
                                                         ne0, ne1, ne2, ne3, circular);
}

void ggml_cuda_op_pad(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    float *             dst_d  = (float *) dst->data;
    cudaStream_t        stream = ctx.stream();

    GGML_TENSOR_UNARY_OP_LOCALS;

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int32_t lp0      = ((const int32_t *) (dst->op_params))[0];
    const int32_t rp0      = ((const int32_t *) (dst->op_params))[1];
    const int32_t lp1      = ((const int32_t *) (dst->op_params))[2];
    const int32_t rp1      = ((const int32_t *) (dst->op_params))[3];
    const int32_t lp2      = ((const int32_t *) (dst->op_params))[4];
    const int32_t rp2      = ((const int32_t *) (dst->op_params))[5];
    const int32_t lp3      = ((const int32_t *) (dst->op_params))[6];
    const int32_t rp3      = ((const int32_t *) (dst->op_params))[7];
    const int32_t circular = ((const int32_t *) (dst->op_params))[8];

    const size_t s00 = nb00 / ggml_type_size(src0->type);
    const size_t s01 = nb01 / ggml_type_size(src0->type);
    const size_t s02 = nb02 / ggml_type_size(src0->type);
    const size_t s03 = nb03 / ggml_type_size(src0->type);

    pad_f32_cuda(src0_d, s00, s01, s02, s03, dst_d,
                 lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3,
                 dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                 (bool) circular, stream);
}
