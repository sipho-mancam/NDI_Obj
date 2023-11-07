#include "ndi_api.hpp"
#include "interface_manager.hpp"
#include "decklink_api.hpp"
#include "streams.hpp"
#include "console_control.hpp"

#include <opencv2/highgui.hpp>

#include <fstream>
#include <iostream>
#include <ctime>
#include <stdio.h>

static IDeckLinkOutput* outDevice;
HRESULT InputLoopThrough(NDI_Recv* input_source);

void log_to_file()
{
    std::ofstream log_file("C:\\Users\\Chroma2\\Documents\\ndi_deck_log.txt");

    auto start = std::chrono::high_resolution_clock::now();
    char buf[256];
    while (true)
    {
      
        if ((std::chrono::high_resolution_clock::now() - start) >= std::chrono::seconds(1))
        {
            time_t curr_time;
            curr_time = time(NULL);

            tm* tm_local = localtime(&curr_time);

            sprintf(buf, "[ %d: %d: %d]: Time log.", tm_local->tm_hour, tm_local->tm_min, tm_local->tm_sec);
            
            log_file << buf << std::endl;
            memset(buf, 0, 256);

            start = std::chrono::high_resolution_clock::now();
        } 
    }
}



int main()
{
    setColor(0x71);
    clrscr();

    init();
    bool exit_flag = false, outputStarted = false;

    DeckLinkCard* card = new DeckLinkCard();
    
    CameraOutputPort* video_out = card->SelectCamOutputPort(2, 0);
    outDevice = video_out->getOutputDevice();
    

    setColor(0x78);
    box(1, 5, 10, 2);
    gotoxy(6, 2);
    printf(" 1. HD");

    box(1, 17, 10, 2);
    gotoxy(18, 2);
    printf(" 2. UHD");

    gotoxy(5, 4);
    printf("\n");
    //printf("Select: \n"); // remove new line character for UHD selection

    ndi_deck::StreamManager stream_manager;
    ndi_deck::OutputStream* out_stream = nullptr;
    int mode;
    //std::cin >> mode;

    mode = 1;

    if (mode != 2)
    {
       out_stream = stream_manager.create_output_stream(ndi_deck::displayResoltion::HD);
    }
    else {
        out_stream = stream_manager.create_output_stream(ndi_deck::displayResoltion::UHD);
    }

    if (out_stream) 
    {
        int offsetY = 3;
        setColor(0x79);
        gotoxy(5, 6+offsetY);
        box(6+offsetY, 5,20, 2);
        gotoxy(6, 7+offsetY);
        printf(" Using %s", mode == 1 ? "HD1080i50" : "UHD2160p50");
        gotoxy(0, 9 + offsetY);
        out_stream->start_stream();
        setColor(0x70);
        getchar();
    }else {
        setColor(0x4f);
        std::cout << "[Error] Could not obtain output stream ..." << std::endl;
        setColor(0x2f);
        std::cout << "[info] Exiting ...." << std::endl;
        return -1;
    }
        
    getchar();
    stream_manager.kill_all_streams();
    setColor(0x7a);
    std::cout << "[info] Stream Killed successfully" << std::endl;
    setColor(0x70);
    return 0;
    //clean_up();
}

IDeckLinkVideoFrame* Interface_Manager::get_key_signal(NDIlib_video_frame_v2_t& ndi_frame, uint *data)
{
    static IDeckLinkMutableVideoFrame* frame_8 = nullptr, * frame_10 = nullptr;
    IDeckLinkVideoConversion* converter = nullptr;
    static void* buffer = nullptr;
    CHECK_DECK_ERROR(GetDeckLinkFrameConverter(&converter));

    extern IDeckLinkOutput* outDevice;

    if (outDevice) {
        if (frame_10 == nullptr)
        {
            CHECK_DECK_ERROR(
                outDevice->CreateVideoFrame(
                ndi_frame.xres,
                ndi_frame.yres,
                (((ndi_frame.xres + 47) / 48) * 128),
                bmdFormat10BitYUV,
                bmdFrameFlagDefault, 
                &frame_10)
            );   
        }

        if (frame_8 == nullptr)
        {
            CHECK_DECK_ERROR(outDevice->CreateVideoFrame(
                ndi_frame.xres,
                ndi_frame.yres,
                (ndi_frame.xres * 16/8),
                bmdFormat8BitYUV,
                bmdFrameFlagDefault, 
                &frame_8)
            );
        }

        frame_8->GetBytes(&buffer);
        memcpy(buffer, data, ndi_frame.xres * 2 * ndi_frame.yres);
        CHECK_DECK_ERROR(converter->ConvertFrame(frame_8, frame_10));
        cudaFreeHost(data);
        return frame_10;
    }
    return nullptr;




}

IDeckLinkVideoFrame* Interface_Manager::convert_ndi_2_decklink_frame_s(NDIlib_video_frame_v2_t* ndi_frame)
{
    // Declare some decklink object and allocate it some memory .... copy video parameters and return it
    IDeckLinkMutableVideoFrame* frame = nullptr, * outframe = nullptr;
    IDeckLinkVideoConversion* converter = nullptr;
    static IDeckLinkMutableVideoFrame* frame_8 = nullptr, *frame_10=nullptr;

    extern IDeckLinkOutput* outDevice;

    if (outDevice) {
        if (frame_10 == nullptr)
        {
            CHECK_DECK_ERROR(outDevice->CreateVideoFrame(
                ndi_frame->xres,
                ndi_frame->yres,
                (((ndi_frame->xres + 47) / 48) * 128),
                bmdFormat10BitYUV,
                bmdFrameFlagDefault, &frame_10));
        }
        
    }

    void* buffer = nullptr;


    CHECK_DECK_ERROR(GetDeckLinkFrameConverter(&converter));

    switch (ndi_frame->FourCC)
    {
    case NDIlib_FourCC_type_BGRA: // 8bit BGRA data out..
    {

        if (outDevice)
        {
            if(frame_8 == nullptr)
            CHECK_DECK_ERROR(outDevice->CreateVideoFrame(
                ndi_frame->xres,
                ndi_frame->yres,
                ndi_frame->line_stride_in_bytes,
                bmdFormat8BitBGRA,
                bmdFrameFlagDefault, &frame_8));
        }
        if (frame_8) {
            frame_8->GetBytes(&buffer);
            memcpy(buffer, ndi_frame->p_data, ndi_frame->yres * ndi_frame->line_stride_in_bytes);
        }

        if (converter)
        {
            converter->ConvertFrame(frame_8, frame_10);
        }
        outframe = frame_10;
        break;
    }

    case NDIlib_FourCC_type_UYVY:
    {
        if (outDevice)
        {
            if(frame_8 ==nullptr)
            CHECK_DECK_ERROR(outDevice->CreateVideoFrame(
                ndi_frame->xres,
                ndi_frame->yres,
                ndi_frame->line_stride_in_bytes,
                bmdFormat8BitYUV,
                bmdFrameFlagDefault, &frame_8));
        }

        if (frame_8) {
            frame_8->GetBytes(&buffer);
            memcpy(buffer, ndi_frame->p_data, ndi_frame->yres * ndi_frame->line_stride_in_bytes);
        }

        if (converter)
        {
            converter->ConvertFrame(frame_8, frame_10);
        }
        outframe = frame_10;
        break;
    }
    }
    //outframe->AddRef();
    return outframe;
}




