#include "decklink_api.hpp"

#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "decklink_kernels.hpp"

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
    assert(cudaSuccess == cudaStatus);
    cudaStatus = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStatus);
}

void VideoFrameCallback::convert_10bit_2_rgb()
{
    cudaError_t cudaStatus;
    const dim3 block(16, 16); // 256 threads per block..
    const dim3 grid(width /(2*block.x), height / block.y);
    unpacked_10bityuv_2_rgb <<< grid, block >> > (
        dst_4,
        rgb_data,
        width, height
        );

  

    cudaStatus = cudaGetLastError();
    assert(cudaStatus == cudaSuccess);
    assert(cudaSuccess == cudaDeviceSynchronize());
    assert(cudaSuccess == cudaMemcpy(rgb_data_h, rgb_data, width * height * sizeof(uchar3), cudaMemcpyDeviceToHost));



}
