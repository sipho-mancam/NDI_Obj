#include "decklinkAPI.hpp"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "decklink_kernels.cuh"

void VideoFrameCallback::unpack_10bit_yuv()
{
    cudaError_t cudaStatus;
    const dim3 block(16, 16); // 256 threads per block..
    const dim3 grid(width / (6 * block.x), height / block.y);

    unpack_10bit_yuv_h <<< grid, block >>> (
        (uint4*)gpuMemory,
        dst_4,
        width, height
        );

    cudaStatus = cudaGetLastError();
    assert(cudaStatus == cudaSuccess);
    assert(cudaSuccess == cudaDeviceSynchronize());
}

