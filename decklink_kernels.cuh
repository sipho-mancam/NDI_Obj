
#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/opencv.hpp>

// host launchers ...
void alpha_2_decklink(long width, long height, uchar* alpha_channel /*Host buffer*/, uint** output /*Host buffer*/);
void get_alpha_channel(long width, long height, uchar* bgra, uchar** alpha_out);


// receiving video from decklink function
__global__ void unpack_10bit_yuv_h(uint4* source, uint4* dst, size_t width, size_t height);
__global__ void unpack_10bit_yuv_f(uint4* source, uint4* dst, size_t width, size_t height);
__global__ void unpack_10bit_rbg(uint4* source, uint4* dst, size_t width, size_t height);

// sending video to decklink functions
__global__ void pack_10bit_yuv(uint4* source, uint4* dst, size_t width, size_t height);
__global__ void pack_8bit_yuv(uint4* source, uint4* dst, size_t width, size_t height);

__global__ void alpha_2_yuyv_pack(uchar* source, uint* dst, size_t width, size_t height);
__global__ void bgra_2_alpha(uchar* bgra, uchar* alpha, long width, long height);




