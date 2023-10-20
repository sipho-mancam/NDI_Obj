#pragma once

#include <queue>
#include <iostream>
#include <thread>
#include <typeinfo>

#include "decklink_api.hpp"
#include "ndi_api.hpp"


class Interface_Manager {
private:
    std::queue<IDeckLinkVideoFrame*> decklink_out_q;
    std::queue<IDeckLinkVideoInputFrame*> decklink_in_q;
    std::queue<NDIlib_video_frame_v2_t*> ndi_in_q;
    std::queue<NDIlib_video_frame_v2_t> ndi_out_q;

    std::queue<IDeckLinkVideoFrame*> frames_buffer;

    std::thread* ndi_processor_worker;
    std::thread* decklink_processer_worker;
    
    bool exit_flag;
    long store_count;


    VideoFrameObj* frame;

    NDIlib_video_frame_v2_t NDI_video_frame_10bit;
    NDIlib_video_frame_v2_t NDI_video_frame_16bit;

    NDIlib_video_frame_v2_t convert_decklink_2_ndi_frame(IDeckLinkVideoInputFrame*);
    IDeckLinkVideoFrame* convert_ndi_2_decklink_frame(NDIlib_video_frame_v2_t*);

    void process_ndi_q_thread();
    void process_decklink_q_thread();
    
    void stop();
    void start();

public:
    Interface_Manager(bool start_a = false);
    ~Interface_Manager(); // the manager will only close when all streams are clear...~Interface_Manager();
    // the manager will only close when all streams are clear...
    bool isRunning() { return exit_flag; }

    void start_ndi();
    void start_decklink();

    template <typename T>
    std::queue<T*>* getQRef(bool out = true); // the bool detects for ndi_queues

    std::queue<IDeckLinkVideoInputFrame*>* getDeckLinkInputQ() { return &decklink_in_q; }
    std::queue<IDeckLinkVideoFrame*>* getDeckLinkOutputQ() { return &decklink_out_q; }
    std::queue<NDIlib_video_frame_v2_t*>* getNDIInputQ() { return &ndi_in_q; }
    std::queue<NDIlib_video_frame_v2_t>* getNDIOutputQ() { return &ndi_out_q; }

};
