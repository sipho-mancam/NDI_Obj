#pragma once

#include <iostream>
#include <chrono>
#include <thread>
#include <csignal>
#include <atomic>
#include <set>
#include <string>
#include <queue>
#include <cassert>
#include <conio.h>
#include <exception>

#ifdef _WIN32
#include <windows.h>
#endif
#ifdef _WIN64
#pragma comment(lib, "Processing.NDI.Lib.x64.lib")
#else
#pragma comment(lib, "Processing.NDI.Lib.x86.lib")
#endif
#define _DEBUG
#ifdef _DEBUG
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#endif
#include <Processing.NDI.Lib.h>
#include "NDI_constants.hpp"
#include "common.hpp"
#include "decklinkAPI.hpp"
#include "decklink_kernels.cuh"
#include "LoopThroughVideoFrame.h"
#include "interface_manager.hpp"

static uint32_t ids = 0;

void init();
void clean_up();
uint32_t idGene();
void resetID();

class NDI_exception : public std::exception {
private:
    const char* const msg;

public:
    NDI_exception(const char* const m)
        :msg(m), exception((const char* const)msg) {}

    const char* what() const override
    {
        return msg;
    }
};

class NDI_Obj {
protected:
    uint32_t id;
    bool* exit;
    uint32_t timeout;
public:
    NDI_Obj(bool* ex = nullptr) : exit(ex), id(-1), timeout(50) { id = idGene(); }
    virtual uint32_t getID() { return id; }
    virtual void run() = 0;
};


class Discovery : public NDI_Obj {
private:
    NDIlib_find_create_t discovery_description;
    NDIlib_find_instance_t disc_instance;
    std::set <std::string> discovered_sources;
    std::set <std::string> saved_sources;
    std::vector<std::string> frozen_list; // we use the frozen list to select devices...
    std::string selected_device;
    bool status, save, running;
    uint32_t current_sources_count;
    NDIlib_source_t* sources;
    std::thread* threadI;
    uint32_t time_out;

public:
    Discovery(bool* controller, NDIlib_find_create_t desc = (NDIlib_find_create_t)NULL, bool autoID = true);
    bool getStatus() { return  status; }
    void disableSave() { this->save = false; saved_sources.clear(); }
    void enableSave() { this->save = true; }
    bool isSourcesReady() { return current_sources_count > 0; }
    void run() override;
    void start();
    void stop();
    void showMeList();
    std::string selectDevice(int s);
    std::string getSelectedDevice() { return selected_device; }
    std::set<std::string> getDiscoveredSources();
    std::set<std::string> getSavedSources() { return this->saved_sources; }
    ~Discovery();

    // TODO: Write the discovered sources to a mysql database for UI to update ...
};



class NDI_Recv : public NDI_Obj {
protected:
    std::string source;
    uint32_t channel;
    NDIlib_source_t s_connect;
    NDIlib_recv_instance_t rec_instance;
    NDIlib_recv_create_v3_t recv_desc;

    NDIlib_framesync_instance_t frames_synchronizer;
    std::thread* receiver_thread;
    uint32_t delay;
    std::queue<NDIlib_video_frame_v2_t*>* frames;
    DeckLinkOutputPort* keyPort;
    DeckLinkOutputPort* fillPort;
    bool connected, running, fillAndKey;
   

    NDIlib_video_frame_v2_t* persFrame;

    using VideoInputArrivedCallback = std::function<void(std::shared_ptr<LoopThroughVideoFrame>, std::shared_ptr<LoopThroughVideoFrame>)>;
    VideoInputArrivedCallback videoArrivedCallback;

    void run() override;
    void splitKeyandFill(cv::Mat& src, cv::Mat& dstA, cv::Mat& dstB /*This must be the alpha channel*/);

public:

    NDI_Recv(bool* controller, uint32_t c = -1, std::string s = "");
    void enableFillAndKey() { this->fillAndKey = true; }
    void disableFillAndKey() { this->fillAndKey = false; }
    void disconnect();
    void setKeyAndFillPorts(DeckLinkOutputPort* f, DeckLinkOutputPort* k);

    void connect(std::string s);
    void subscribe_to_q(std::queue<NDIlib_video_frame_v2_t*>* qu);
    void start();
    void stop();

    void onVideoInputArrived(const VideoInputArrivedCallback& callback) { videoArrivedCallback = callback; }

    void popFrame();
    void clearAll();
    uint32_t getChannel() { return channel; }
    std::string getSource() { return source; }

    ~NDI_Recv();
};



class NDI_Sender : public NDI_Obj {
private:
    NDIlib_send_instance_t sender;
    NDIlib_send_create_t desc;
    NDIlib_video_frame_v2_t NDI_video_frame_10bit;
    std::queue<NDIlib_video_frame_v2_t>* frames_q;
    std::thread* p_worker;
    bool init_d, running;
    NDIlib_video_frame_v2_t NDI_video_frame_16bit;
    std::mutex mtx;

public:
    NDI_Sender(bool* controller, std::string source = "");
    
    void start();
    void stop();
    bool isRunning() { return running; }
    void run() override;
    void subscribe_to_q(std::queue<NDIlib_video_frame_v2_t>* q);
    ~NDI_Sender();
};


class NDI_Key_And_Fill : public NDI_Recv {
private:

    void run() override;

public:

    NDI_Key_And_Fill(bool* controller, uint32_t c = -1, std::string s = "");
    void enableFillAndKey() { this->fillAndKey = true; }
    void disableFillAndKey() { this->fillAndKey = false; }
    void setKeyAndFillPorts(DeckLinkOutputPort* f, DeckLinkOutputPort* k);

    ~NDI_Key_And_Fill();

};