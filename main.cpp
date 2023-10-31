#include "ndi_api.hpp"
#include "interface_manager.hpp"
#include "decklink_api.hpp"
#include "streams.hpp"


#include <fstream>
#include <iostream>
#include <ctime>

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
    init();
    bool exit_flag = false, outputStarted = false;

    Interface_Manager interface_manager;
    interface_manager.start_decklink();

    DeckLinkCard* card = new DeckLinkCard();
    DeckLinkInputPort* inputPort = card->SelectInputPort(0);
    assert(inputPort != nullptr);
    inputPort->subscribe_2_input_q(interface_manager.getDeckLinkInputQ());
    inputPort->startCapture();
    
    CameraOutputPort* video_out = card->SelectCamOutputPort(2, 0);
    outDevice = video_out->getOutputDevice();
    
    NDI_Sender* sender = new NDI_Sender(&exit_flag, "");
    sender->subscribe_to_q(interface_manager.getNDIOutputQ());
    sender->start();

    ndi_deck::StreamManager stream_manager;
    ndi_deck::OutputStream* out_stream = stream_manager.create_output_stream();
    out_stream->start_stream();
    getchar();
    stream_manager.kill_all_streams();
    std::cout << "[info] Stream Killed successfully" << std::endl;
    getchar();

    clean_up();
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




