#pragma once

#include <iostream>
#include <string>
#include <unordered_map>


#include <Processing.NDI.Lib.h>

class NDIVideoFrame {
private:
	long width, height, rowBytes;
	std::string pixelFormat;
	float frameRate;
	std::string videoType;

	void *data;
	


public:

	std::unordered_map<long, std::string> ndi2String_type;
	std::unordered_map<uint32_t, std::string> ndi2String_pixelFormat;

	NDIVideoFrame(NDIlib_video_frame_v2_t* frame)
		: width(frame->xres), height(frame->yres), rowBytes(frame->line_stride_in_bytes)
	{
		init();
		pixelFormat = ndi2String_pixelFormat[frame->FourCC];
		frameRate = frame->frame_rate_N / (frame->frame_rate_D*1.0);
		videoType = ndi2String_type[frame->frame_format_type];
		data = frame->p_data;
	}

	NDIVideoFrame()
		: width(0), height(0), rowBytes(0)
	{
		NDIVideoFrame::init();
		pixelFormat = "";
		frameRate = 0;
		videoType = "";
		data = NULL;
	}

	void printParams()
	{
		printf("Params: \n---------------\n");
		printf("Frame Type:\t%s\n", videoType.c_str());
		printf("Pixel Format:\t%s\n", pixelFormat.c_str());
		printf("Resolution:\t%d x %d\n", width, height);
		printf("Frame Rate:\t%.2f fps\n", frameRate);
		printf("Data:\t%p\n", data);
	}

	void AddFrame(NDIlib_video_frame_v2_t* frame)
	{
		width = frame->xres;
		height = frame->yres;
		rowBytes = frame->line_stride_in_bytes;

		pixelFormat = ndi2String_pixelFormat[frame->FourCC];
		frameRate = frame->frame_rate_N / (frame->frame_rate_D * 1.0);
		videoType = ndi2String_type[frame->frame_format_type];
		data = frame->p_data;
	}

	void init()
	{
		ndi2String_type[NDIlib_frame_format_type_progressive] = "progressive";
		ndi2String_type[NDIlib_frame_format_type_interleaved] = "interleaved";

		ndi2String_pixelFormat[NDIlib_FourCC_video_type_RGBA] = "RGBA";
		ndi2String_pixelFormat[NDIlib_FourCC_video_type_BGRA] = "BGRA";
		ndi2String_pixelFormat[NDIlib_FourCC_video_type_UYVY] = "UYVY";
		ndi2String_pixelFormat[NDIlib_FourCC_video_type_UYVA] = "UYVA";
		ndi2String_pixelFormat[NDIlib_FourCC_video_type_BGRX] = "BGRX";
		ndi2String_pixelFormat[NDIlib_FourCC_video_type_RGBX] = "RGBX";
	}



};