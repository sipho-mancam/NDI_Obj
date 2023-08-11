
#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


// receiving video from decklink function
__global__ void unpack_10bit_yuv(uint4* source, uint4* dst, size_t width, size_t height);
__global__ void unpack_10bit_rbg(uint4* source, uint4* dst, size_t width, size_t height);

// sending video to decklink functions
__global__ void pack_10bit_yuv(uint4* source, uint4* dst, size_t width, size_t height);
__global__ void pack_8bit_yuv(uint4* source, uint4* dst, size_t width, size_t height);

