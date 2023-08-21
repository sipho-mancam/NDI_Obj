
#include "decklink_kernels.cuh"
#include <opencv2/opencv.hpp>

inline __device__ __host__ double clamp(double f, double a, double b)
{
	return (double)fmaxf(a, fminf(f, b));
}

// unpack to half ....
__global__ void unpack_10bit_yuv_h(uint4* source, uint4* dst, size_t width, size_t height)
{
	// width is the original image width ... 1920 or 3840 

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	int srcWidth = width / 6, dstWidth = width / 2; 


	uint4* macroPx;
	macroPx = &source[y * srcWidth + x];
	
	double Cr0, Y0, Cb0, Y1, Cb2, Y2, Cr2, Y3, Cb4, Y5, Cr4, Y4;

	Cb0 = (macroPx->x & 0x3ff);
	Y0 = ((macroPx->x & 0xffc00) >> 10);

	Cr0 = (macroPx->x >> 20);
	Y1 = (macroPx->y & 0x3ff);
	
	Cb2 = ((macroPx->y & 0xffc00) >> 10);
	Y2 = (macroPx->y >> 20);

	Cr2 = (macroPx->z & 0x3ff);
	Y3 = ((macroPx->z & 0xffc00) >> 10);
	
	Cb4 = (macroPx->z >> 20);
	Y4 = (macroPx->w & 0x3ff);

	Cr4 = ((macroPx->w & 0xffc00) >> 10);
	Y5 = (macroPx->w >> 20);
	
	dst[y * dstWidth + (x * 3) + 0] = make_uint4(Cr0, Y0, Cb0, Y1); // x y z w
	dst[y * dstWidth + (x * 3) + 1] = make_uint4(Cr2, Y2, Cb2, Y3);
	dst[y * dstWidth + (x * 3) + 2] = make_uint4(Cr4, Y4, Cb4, Y5);
}


__global__ void unpacked_10bityuv_2_rgb(uint4* src, uchar3* dst, int width, int height)
{

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;
	
	int srcWidth = width / 2;

	uint4* macroPx;
	macroPx = &src[y * srcWidth + x];

	double3 px_0 = make_double3(
		clamp(macroPx->y + 1.540f * (macroPx->z - 512.0), 0.0, 1023.0),
		clamp(macroPx->y - 0.459f * (macroPx->x - 512.0) - 0.183f * (macroPx->z - 512.0), 0.0, 1023.0),
		clamp(macroPx->y + 1.816f * (macroPx->x - 512.0), 0.0, 1023.0));

	double3	px_1 = make_double3(
		clamp(macroPx->w + 1.540f * (macroPx->z - 512.0), 0.0, 1023.0),
		clamp(macroPx->w - 0.459f * (macroPx->x - 512.0) - 0.183f * (macroPx->z - 512.0), 0.0, 1023.0),
		clamp(macroPx->w + 1.816f * (macroPx->x - 512.0), 0.0, 1023.0));

	dst[y * width + (x * 2) + 0].x = clamp(px_0.x / 1024.0 * 255.0, 0.0f, 255.0f);
	dst[y * width + (x * 2) + 0].y = clamp(px_0.y / 1024.0 * 255.0, 0.0f, 255.0f);
	dst[y * width + (x * 2) + 0].z = clamp(px_0.z / 1024.0 * 255.0, 0.0f, 255.0f);

	dst[y * width + (x * 2) + 1].x = clamp(px_1.x / 1024.0 * 255.0, 0.0f, 255.0f);
	dst[y * width + (x * 2) + 1].y = clamp(px_1.y / 1024.0 * 255.0, 0.0f, 255.0f);
	dst[y * width + (x * 2) + 1].z = clamp(px_1.z / 1024.0 * 255.0, 0.0f, 255.0f);
}


__global__ void unpack_10bit_yuv_f(uint4* source, uint* dst, size_t width, size_t height)
{
	// width is the original image width ... 1920 or 3840 
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	int srcWidth = width / 6;

	if (x >= srcWidth || y >= height)
		return;

	uint4* macroPx;
	macroPx = &source[y * srcWidth + x];

	double Cr0 ,Y0, Cb0, Y1, Cb2, Y2, Cr2, Y3, Cb4, Y5, Cr4, Y4;

	Cb0 = (macroPx->x & 0x3ff);
	Y0 = ((macroPx->x & 0xffc00) >> 10);
	Cr0 = (macroPx->x >> 20);

	Y1 = (macroPx->y & 0x3ff);

	Cb2 = ((macroPx->y & 0xffc00) >> 10);
	Y2 = (macroPx->y >> 20);
	Cr2 = (macroPx->z & 0x3ff);

	Y3 = ((macroPx->z & 0xffc00) >> 10);

	Cb4 = (macroPx->z >> 20);
	Y4 = (macroPx->w & 0x3ff);
	Cr4 = ((macroPx->w & 0xffc00) >> 10);
	
	Y5 = (macroPx->w >> 20);

	dst[y * width + (x * 18) + 0] = Cb0;
	dst[y * width + (x * 18) + 1] = Y0;
	dst[y * width + (x * 18) + 2] = Cr0;

	dst[y * width + (x * 18) + 3] = 0;
	dst[y * width + (x * 18) + 4] = Y1;
	dst[y * width + (x * 18) + 5] = 0;

	dst[y * width + (x * 18) + 6] = Cb2;
	dst[y * width + (x * 18) + 7] = Y2;
	dst[y * width + (x * 18) + 8] = Cr2;

	dst[y * width + (x * 18) + 9] = 0;
	dst[y * width + (x * 18) + 10] = Y3;
	dst[y * width + (x * 18) + 11] = 0;

	dst[y * width + (x * 18) + 12] = Cb4;
	dst[y * width + (x * 18) + 13] = Y4;
	dst[y * width + (x * 18) + 14] = Cr4;

	dst[y * width + (x * 18) + 15] = 0;
	dst[y * width + (x * 18) + 16] = Y5;
	dst[y * width + (x * 18) + 17] = 0;
}


__global__ void unpack_10bit_rbg(uchar* source, uint4* dst, size_t width, size_t height)
{

}

__global__ void pack_10bit_yuv(uint4* source, uint4* dst, size_t width, size_t height)
{
	

}

// pack 10bit yuv --> yu yv --> to pixels per word
__global__ void pack_8bit_yuv(uchar* source, uint* dst, size_t width, size_t height)
{

}

void alpha_2_decklink(long width, long height, uchar *alpha_channel /*Host buffer*/, uint** output)
{

	cudaError_t cudaStatus;
	const dim3 block(16, 16); // 256 threads per block..

	int rows = height / block.y;
	if (height % block.y > 0)
		rows += block.y - height % block.y;

	const dim3 grid(width / (2 * block.x), rows);

	uchar* pinnedBuf, *in_gpuBuf;
	uint* gpuBuf_out;
	uint* cpuOut;

	size_t packedSize = (width / 2) * height * sizeof(uint);

	assert(cudaSuccess == cudaMallocHost((void**)&pinnedBuf, width * height*sizeof(uchar)));
	assert(cudaSuccess == cudaMallocHost((void**)&cpuOut, packedSize));

	memcpy(pinnedBuf, alpha_channel, width*height); // single channel copy

	assert(cudaSuccess == cudaMalloc((void**)&in_gpuBuf, width * height));
	assert(cudaSuccess == cudaMemcpy((void*)in_gpuBuf, (void*)pinnedBuf, width * height, cudaMemcpyHostToDevice));
	assert(cudaMalloc((void**)&gpuBuf_out, packedSize) == cudaSuccess);
	
	alpha_2_yuyv_pack <<< grid, block >>> (
		in_gpuBuf,
		gpuBuf_out,
		width, height
		);

	cudaStatus = cudaGetLastError();
	assert(cudaStatus == cudaSuccess);
	assert(cudaSuccess == cudaDeviceSynchronize());

	cudaStatus = cudaMemcpy(cpuOut, gpuBuf_out, packedSize, cudaMemcpyDeviceToHost);
	assert(cudaStatus == cudaSuccess);
	*output = cpuOut;

	cudaFreeHost(pinnedBuf);
	cudaFree(in_gpuBuf);
	cudaFree(gpuBuf_out);
}


void alpha_2_decklink_gpu(long width, long height, uchar* alpha_channel /*GPU Buffer*/, uint** output /*CPU Buffer*/)
{

	cudaError_t cudaStatus;
	const dim3 block(16, 16); // 256 threads per block..
	const dim3 grid(width / (2 * block.x), round(height+8 / block.y));

	uint* gpuBuf_out;
	uint* cpuOut;

	size_t packedSize = (width / 2) * height * sizeof(uint);

	assert(cudaSuccess == cudaMallocHost((void**)&cpuOut, packedSize));
	assert(cudaMalloc((void**)&gpuBuf_out, packedSize) == cudaSuccess);

	alpha_2_yuyv_pack <<< grid, block >>> (
		alpha_channel,
		gpuBuf_out,
		width, height
		);

	cudaStatus = cudaGetLastError();
	assert(cudaStatus == cudaSuccess);
	assert(cudaSuccess == cudaDeviceSynchronize());

	cudaStatus = cudaMemcpy(cpuOut, gpuBuf_out, packedSize, cudaMemcpyDeviceToHost);
	assert(cudaStatus == cudaSuccess);

	*output = cpuOut;

	cudaFree(alpha_channel);
	cudaFree(gpuBuf_out);
}

__global__ void alpha_2_yuyv_pack(uchar* source, uint* dst, size_t width, size_t height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	int dstWidth = width / 2;

	uchar Y0, Y1, Cb, Cr;

	Y0 = source[y * width + (x * 2)];
	Y1 = source[y * width + (x * 2) + 1];
	Cb = -127;
	Cr = -127;

	dst[y * dstWidth + x] |= (uint)(Y1 << 24);
	dst[y * dstWidth + x] |= (uint)(Cr << 16);
	dst[y * dstWidth + x] |= (uint)(Y0 << 8);
	dst[y * dstWidth + x] |= (uint)(Cb);

}

void get_alpha_channel(long width, long height, uchar* bgra, uchar** alpha_out)
{
	cudaError_t cudaStatus;
	const dim3 block(16, 16); // 256 threads per block..
	const dim3 grid(width / block.x, height / block.y);

	uchar * in_gpu_buf; // bgra pinned and gpu buffers
	uchar* pinned_alpha, *out_alpha;

	size_t bgra_size = width * height * 4;
	size_t alpha_size = width * height * 1;

	assert(cudaSuccess == cudaMalloc((void**)&in_gpu_buf, bgra_size));
	assert(cudaSuccess == cudaMemcpy(in_gpu_buf, bgra, bgra_size, cudaMemcpyHostToDevice));
	// BGRA data is now in device memory. ...
	assert(cudaSuccess == cudaMalloc((void**)&out_alpha, alpha_size));

	bgra_2_alpha <<<grid, block >>> (
		in_gpu_buf,
		out_alpha,
		width, height
		);

	cudaStatus = cudaGetLastError();
	assert(cudaStatus == cudaSuccess);
	assert(cudaSuccess == cudaDeviceSynchronize());

	assert(cudaSuccess == cudaMallocHost((void**)&pinned_alpha, alpha_size));
	assert(cudaSuccess == cudaMemcpy(pinned_alpha, out_alpha, alpha_size, cudaMemcpyDeviceToHost));

	*alpha_out = pinned_alpha;

	cudaFree(in_gpu_buf);
	cudaFree(out_alpha);
}


void get_alpha_channel_gpu(long width, long height, uchar* bgra /*GPU buffer*/, uchar** alpha_out)
{
	cudaError_t cudaStatus;
	const dim3 block(16, 16); // 256 threads per block..
	const dim3 grid(width / block.x, height / block.y);

	uchar * out_alpha;
	size_t alpha_size = width * height;

	assert(cudaSuccess == cudaMalloc((void**)&out_alpha, alpha_size));

	bgra_2_alpha << <grid, block >> > (
		bgra,
		out_alpha,
		width, height
		);

	cudaStatus = cudaGetLastError();
	assert(cudaStatus == cudaSuccess);
	assert(cudaSuccess == cudaDeviceSynchronize());

	*alpha_out = out_alpha;

	//cudaFree(in_gpu_buf);
	cudaFree(bgra);

}

__global__ void bgra_2_alpha(uchar* bgra, uchar* alpha, long width, long height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	int numChannels = 4;
	int selectedChannel = 3;

	alpha[y * width + x] = bgra[y * (width * numChannels) + (x * numChannels)+ selectedChannel];
}

uchar* get_yuv_from_bgr_packed(long width, long height, uchar* bgra, uint** output)
{
	cudaError_t cudaStatus;
	const dim3 block(16, 16); // 256 threads per block..
	const dim3 grid(width / (2*block.x), height / block.y);

	uchar* in_gpu_buf; // bgra pinned and gpu buffers
	uint* pinned_yuv, * out_yuv;

	size_t bgra_size = width * height * 4;
	size_t yuv_size = sizeof(uint) * (width / 2) * height;

	assert(cudaSuccess == cudaMalloc((void**)&in_gpu_buf, bgra_size));
	assert(cudaSuccess == cudaMemcpy(in_gpu_buf, bgra, bgra_size, cudaMemcpyHostToDevice));

	// BGRA data is now in device memory. ...
	assert(cudaSuccess == cudaMalloc((void**)&out_yuv, yuv_size));
	assert(cudaSuccess == cudaMallocHost((void**)&pinned_yuv, yuv_size));


	bgr_2_8bityuv_packed << <grid, block >> > (
		in_gpu_buf,
		out_yuv,
		width, height
		);

	cudaStatus = cudaGetLastError();
	assert(cudaStatus == cudaSuccess);
	assert(cudaSuccess == cudaDeviceSynchronize());

	assert(cudaSuccess == cudaMemcpy(pinned_yuv, out_yuv, yuv_size, cudaMemcpyDeviceToHost));
	*output = pinned_yuv;

	cudaFree(out_yuv);

	return in_gpu_buf;
}


__global__ void bgr_2_8bityuv_packed(uchar* bgra, uint* dst, long width, long height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	int dstWidth = width / 2;
	int numChannels = 4;

	uchar Y0, Y1, Cb, Cr;

	uchar R1, R2, G1, G2, B1, B2;

	B1 = bgra[y * (width * numChannels) + ((2 * x) * numChannels) + 0];
	G1 = bgra[y * (width * numChannels) + ((2 * x) * numChannels) + 1];
	R1 = bgra[y * (width * numChannels) + ((2 * x) * numChannels) + 2];


	B2 = bgra[y * (width * numChannels) + (((2 * x) + 1) * numChannels) + 0];
	G2 = bgra[y * (width * numChannels) + (((2 * x) + 1) * numChannels) + 1];
	R2 = bgra[y * (width * numChannels) + (((2 * x) + 1) * numChannels) + 2];


	// convert to YUV color space ....
	Y0 = (0.299 * R1 + 0.587 * G1 + 0.114 * B1);

	Cb = 0.492 * (B1 - Y0);//(-0.299 * R1 - 0.587 * G1 + 0.886 * B1);
	Cr = 0.877 * (R1 - Y0);//-127;// (0.701 * R1 - 0.587 * G1 - 0.114 * B1);

	Y1 = (0.299 * R2 + 0.587 * G2 + 0.114 * B2);
	

	// pack it for decklink ....
	dst[y * dstWidth + x] |= (uint)(Y1 << 24);
	dst[y * dstWidth + x] |= (uint)(Cr << 16);
	dst[y * dstWidth + x] |= (uint)(Y0 << 8);
	dst[y * dstWidth + x] |= (uint)(Cb);
}


__global__ void pack_yuv_10bit(uint4* packed_Video, uint4* unpacked_video, int srcAlignedWidth, int dstAlignedWidth, int height)
{

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= srcAlignedWidth || y >= height)
		return;


	uint4* macroPx;
	macroPx = &packed_Video[y * srcAlignedWidth + x];
	double Cr0;
	double Y0;
	double Cb0;

	double Y2;
	double Cb2;
	double Y1;

	double Cb4;
	double Y3;
	double Cr2;

	double Y5;
	double Cr4;
	double Y4;

	Cr0 = unpacked_video[y * dstAlignedWidth + (x * 3) + 0].x;
	Y0 = unpacked_video[y * dstAlignedWidth + (x * 3) + 0].y;
	Cb0 = unpacked_video[y * dstAlignedWidth + (x * 3) + 0].z;
	Y1 = unpacked_video[y * dstAlignedWidth + (x * 3) + 0].w;

	Cb2 = unpacked_video[y * dstAlignedWidth + (x * 3) + 1].z;;
	Y2 = unpacked_video[y * dstAlignedWidth + (x * 3) + 1].y;
	Cb4 = unpacked_video[y * dstAlignedWidth + (x * 3) + 2].z;
	Y3 = unpacked_video[y * dstAlignedWidth + (x * 3) + 1].w;

	Cr2 = unpacked_video[y * dstAlignedWidth + (x * 3) + 1].x;
	Y4 = unpacked_video[y * dstAlignedWidth + (x * 3) + 2].y;
	Cr4 = unpacked_video[y * dstAlignedWidth + (x * 3) + 2].x;
	Y5 = unpacked_video[y * dstAlignedWidth + (x * 3) + 2].w;


	macroPx->x = ((unsigned int)Cr0 << 20) + ((unsigned int)Y0 << 10) + (unsigned int)Cb0;
	macroPx->y = macroPx->y & 0x3ffffc00;
	macroPx->y = macroPx->y | (unsigned int)Y1;

	macroPx->y = macroPx->y & 0x3ff;
	macroPx->y = macroPx->y | ((unsigned int)Y2 << 20) | ((unsigned int)Cb2 << 10);
	macroPx->z = macroPx->z & 0x3ff00000;
	macroPx->z = macroPx->z | (((unsigned int)Y3 << 10) | (unsigned int)Cr2);

	macroPx->z = macroPx->z & 0xfffff;
	macroPx->z = macroPx->z | ((unsigned int)Cb4 << 20);
	macroPx->w = ((long)Y5 << 20) + ((unsigned int)Cr4 << 10) + (unsigned int)Y4;

}