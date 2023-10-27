#include "interface_manager.hpp"
#include <thread>

template <typename T>
std::queue<T*>* Interface_Manager::getQRef(bool out)
{
    std::cout << "It runs" << std::endl;
    // passed type 
    std::string type(typeid(T).name());

    std::string decklink_out(typeid(IDeckLinkVideoFrame).name());
    std::string decklink_in(typeid(IDeckLinkVideoInputFrame).name());
    std::string ndi_type(typeid(NDIlib_video_frame_v2_t).name());

    if (type == decklink_in)
    {
        std::cout << "We are giving you the input Q" << std::endl;
        _getch();
        return &decklink_in_q;
    }
    if (type == decklink_out)
    {
        return &decklink_out_q;
    }
    if (type == ndi_type)
    {
        if (out)
            return &ndi_out_q;
        return &ndi_in_q;
    }

    return nullptr;
}


void Interface_Manager::process_decklink_q_thread()
{
    // this function processes the decklink input q and converts it to NDI frame
    while (!exit_flag)
    {
        if (!decklink_in_q.empty())
        {
            IDeckLinkVideoInputFrame* frame = decklink_in_q.front();
            NDIlib_video_frame_v2_t ndi_frame = convert_decklink_2_ndi_frame(frame);
            decklink_in_q.pop(); // manage the queue memory
            ndi_out_q.push(ndi_frame);
        }
    }
}

void Interface_Manager::process_ndi_q_thread()
{
    // this function processes the ndi input frame and converts it to DeckLinkFrame
    while (!exit_flag)
    {
        if (!ndi_in_q.empty())
        {
            NDIlib_video_frame_v2_t* frame = ndi_in_q.front();
            IDeckLinkVideoFrame* deck_frame = convert_ndi_2_decklink_frame(frame);
            ndi_in_q.pop();
            decklink_out_q.push(deck_frame);
        }
    }
}


IDeckLinkVideoFrame* Interface_Manager::convert_ndi_2_decklink_frame(NDIlib_video_frame_v2_t* ndi_frame)
{
    // Declare some decklink object and allocate it some memory .... copy video parameters and return it
    VideoFrameObj* outframe = nullptr;
   
    switch (ndi_frame->FourCC)
    {
    case NDIlib_FourCC_type_BGRA: // 8bit BGRA data out..
    {
        if(frame == nullptr)
            frame = new VideoFrameObj(ndi_frame->xres, ndi_frame->yres, bmdFormat8BitBGRA);
        
        outframe = frame;
        outframe->SetFrameData(ndi_frame->p_data);
        break;
    }

    case NDIlib_FourCC_type_UYVY:
    {
        if (frame == nullptr)
            frame = new VideoFrameObj(ndi_frame->xres, ndi_frame->yres, bmdFormat8BitYUV);
        outframe = frame;
        outframe->SetFrameData(ndi_frame->p_data);
        break;
    }
    }
    outframe = frame;

    return outframe;
}


NDIlib_video_frame_v2_t Interface_Manager::convert_decklink_2_ndi_frame(IDeckLinkVideoInputFrame* frame)
{
    // declare the NDI frame, copy the decklink parameters over to it, ... and return it.
    
    try {
        switch (frame->GetPixelFormat())
        {
        case bmdFormat10BitYUV:
        {
            NDI_video_frame_10bit.xres = frame->GetWidth();
            NDI_video_frame_10bit.yres = frame->GetHeight();
            NDI_video_frame_10bit.FourCC = (NDIlib_FourCC_video_type_e)NDI_LIB_FOURCC('V', '2', '1', '0');
            NDI_video_frame_10bit.line_stride_in_bytes = frame->GetRowBytes();
            NDI_video_frame_10bit.frame_rate_N = frame->GetWidth() == 3840 ? 50000: 25000;
            NDI_video_frame_10bit.frame_rate_D = 1000;
            NDI_video_frame_10bit.frame_format_type = frame->GetWidth() == 1920 ?NDIlib_frame_format_type_interleaved : NDIlib_frame_format_type_progressive;
            NDI_video_frame_10bit.picture_aspect_ratio = 16.0f / 9.0f;
            NDI_video_frame_10bit.timecode = NDIlib_send_timecode_synthesize;
            uchar* buf;

            frame->GetBytes((void**)&buf);
            if(!NDI_video_frame_10bit.p_data)
                NDI_video_frame_10bit.p_data = (uint8_t*)malloc(frame->GetRowBytes() * frame->GetHeight());

            memcpy(NDI_video_frame_10bit.p_data, buf, NDI_video_frame_10bit.line_stride_in_bytes * NDI_video_frame_10bit.yres);

            NDI_video_frame_16bit.xres = NDI_video_frame_10bit.xres;
            NDI_video_frame_16bit.yres = NDI_video_frame_10bit.yres;

            NDI_video_frame_16bit.line_stride_in_bytes = NDI_video_frame_16bit.xres * sizeof(uint16_t);
            if(!NDI_video_frame_16bit.p_data)
                NDI_video_frame_16bit.p_data = (uint8_t*)malloc(NDI_video_frame_16bit.line_stride_in_bytes * 2 * NDI_video_frame_16bit.yres);
            // Convert into the destination
            NDIlib_util_V210_to_P216(&NDI_video_frame_10bit, &NDI_video_frame_16bit);
            break;
        }

        }
    }
    catch (std::exception& re) {
        std::cout << re.what() << std::endl;
    }

    frame->Release();
    
    return NDI_video_frame_16bit;
}

void Interface_Manager::start()
{
    // set exit flag to false;
    /*
    * 1. set exit flag to false ...
    * 2. create threads for ndi_processing and decklink_processing.
    * 3.
    */
    exit_flag = false;
    decklink_processer_worker = new std::thread(&Interface_Manager::process_decklink_q_thread, this);
    ndi_processor_worker = new std::thread(&Interface_Manager::process_ndi_q_thread, this);

    return;
}

void Interface_Manager::start_ndi()
{
    if (ndi_processor_worker)
    {
        ndi_processor_worker->join();
        delete ndi_processor_worker;
    }
    ndi_processor_worker = new std::thread(&Interface_Manager::process_ndi_q_thread, this);
}

void Interface_Manager::start_decklink()
{
    if (decklink_processer_worker)
    {
        decklink_processer_worker->join();
        delete decklink_processer_worker;
    }
    decklink_processer_worker = new std::thread(&Interface_Manager::process_decklink_q_thread, this);
}

void Interface_Manager::stop()
{
    /*
    * 1. set exit flag to false;
    * 2. join threads ...
    * 4. wait for queues to be empty, then exit ...
    */
    exit_flag = true; // this will stop the output queues from being added...

    if (ndi_processor_worker)
    {
        ndi_processor_worker->join();
        delete ndi_processor_worker;
        ndi_processor_worker = nullptr;
    }

    if (decklink_processer_worker)
    {
        decklink_processer_worker->join();
        delete decklink_processer_worker;
        decklink_processer_worker = nullptr;
    }

    std::cout << "Waiting for the streams to wrap up ..." << std::endl;
    // wait for output queues to be empty ...
    auto start = std::chrono::high_resolution_clock::now();
    while ((!ndi_out_q.empty() || !decklink_out_q.empty()) &&
        ((std::chrono::high_resolution_clock::now() - start) < std::chrono::seconds(5)) )
    {}

    return;
}

Interface_Manager::~Interface_Manager()
{
    stop();
}

Interface_Manager::Interface_Manager(bool start_a)
    : store_count(2), 
    decklink_processer_worker(nullptr), 
    ndi_processor_worker(nullptr),
    exit_flag(false), frame(nullptr)
{
    if (start_a)
        this->start();
}