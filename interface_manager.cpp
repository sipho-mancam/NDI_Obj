#include "interface_manager.hpp"
#include <thread>



template <typename T>
std::queue<T>* Interface_Manager::getQRef(bool out)
{
    // passed type 
    std::string type(typeid(T).name());

    std::string decklink_out(typeid(IDeckLinkVideoFrame).name());
    std::string decklink_in(typeid(IDeckLinkVideoInputFrame).name());
    std::string ndi_type(typeid(NDIlib_video_frame_v2_t));

    if (type == decklink_in)
    {
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
            decklink_in_q.pop(); // manage the queue memory
            NDIlib_video_frame_v2_t ndi_frame = convert_decklink_2_ndi_frame(frame);
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
            NDIlib_video_frame_v2_t frame = ndi_in_q.front();
            ndi_in_q.pop();
            IDeckLinkVideoFrame* deck_frame = convert_ndi_2_decklink_frame(frame);
            decklink_out_q.push(deck_frame);
        }
    }
}


IDeckLinkVideoFrame* Interface_Manager::convert_ndi_2_decklink_frame(NDIlib_video_frame_v2_t ndi_frame)
{
    // Declare some decklink object and allocate it some memory .... copy video parameters and return it

    return nullptr;
}

NDIlib_video_frame_v2_t Interface_Manager::convert_decklink_2_ndi_frame(IDeckLinkVideoInputFrame* deck_frame)
{
    // declare the NDI frame, copy the decklink parameters over to it, ... and return it.
    NDIlib_video_frame_v2_t frame;
    return frame;
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
    ndi_processor_worker = new std::thread(&Interface_Manager::process_ndi_q_thread, this);
    decklink_processer_worker = new std::thread(&Interface_Manager::process_decklink_q_thread, this);

    return;

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
    while (!ndi_out_q.empty() || !decklink_out_q.empty()) {}

    return;
}

Interface_Manager::~Interface_Manager()
{
    stop();
}

Interface_Manager::Interface_Manager()
{
    start();
}