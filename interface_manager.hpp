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
    std::queue<NDIlib_video_frame_v2_t> ndi_in_q;
    std::queue<NDIlib_video_frame_v2_t> ndi_out_q;

    std::thread* ndi_processor_worker;
    std::thread* decklink_processer_worker;

    bool exit_flag;

    NDIlib_video_frame_v2_t convert_decklink_2_ndi_frame(IDeckLinkVideoInputFrame*);
    IDeckLinkVideoFrame* convert_ndi_2_decklink_frame(NDIlib_video_frame_v2_t);

    void process_ndi_q_thread();
    void process_decklink_q_thread();
    void start();
    void stop();

public:
    Interface_Manager();
    ~Interface_Manager(); // the manager will only close when all streams are clear...~Interface_Manager();
    // the manager will only close when all streams are clear...
    bool isRunning() { return exit_flag; }

    template <typename T>
    std::queue<T>* getQRef(bool out = true); // the bool detects for ndi_queues

};
