#pragma once

#include <iostream>
#include <list>
#include <opencv2/opencv.hpp>
#include <vector>
#include <conio.h>
#include <cassert>
#include <queue>
#include <exception>
#include <unordered_map>
#include <mutex>

#include <DecklinkAPI_h.h>
#include "platform.h"
#include "DeckLinkDevice.h"

#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"


class DeckLinkCard;
class DeckLinkPlaybackCallback;


class VideoFrameCallback : public FrameArrivedCallback {
private:
    std::queue<IDeckLinkVideoInputFrame*> frames_queue;
    bool droppedFrames, init;
    int maxFrameCount;
    uint32_t* pinnedMemory;
    uint32_t* gpuMemory;
    uint4* dst_4;
    uint* dst_full;
    uchar* buffer;
    BMDPixelFormat pxFormat;
    unsigned int width, height;
    uchar3* rgb_data, * rgb_data_h;
    std::mutex mtx;

public:

    VideoFrameCallback(int mFrameCount = 5);
    void arrived(IDeckLinkVideoInputFrame* frame) override;
    std::queue<IDeckLinkVideoInputFrame*>* getQueRef() { return(&frames_queue); }
    void convert_10bit_2_rgb(); //cuda_function
    void unpack_10bit_yuv(); // cuda_function
    // queue management 
    void clearAll();
    IDeckLinkVideoInputFrame* getFrame();
    IDeckLinkVideoInputFrame* getFrameNoPop();

    size_t queueSize() const { return frames_queue.size(); }
    void popTop() { frames_queue.pop(); }
    bool empty() const { return frames_queue.empty(); }
    bool frameDropped() { return droppedFrames; }
    bool overflow() { return frames_queue.size() == maxFrameCount; }
};

class DeckLinkObject
{
protected:
    HRESULT result;
    HRESULT checkError(std::string info = "", bool fatal = false);
};

class ICallback : public DeckLinkObject
{

};

class IDeckLinkPort : protected DeckLinkObject
{
protected:
    DeckLinkCard* parent;
    IDeckLink* port;
    IDeckLinkDisplayModeIterator* displayModeIterator;
    IDeckLinkDisplayMode* displayMode;
    std::vector<IDeckLinkDisplayMode*> displayModes;
    IDeckLinkProfileAttributes* profileAttributes;
    BMDPixelFormat pixelFormat;
    uint selectedMode;

    virtual void enableVideo() = 0;
    virtual bool doesSupportVideoMode() = 0;
    virtual void configure() = 0;

    IDeckLinkPort(DeckLinkCard* par, IDeckLink* po);
    void SetDisplayMode(int c);
    void SetPixelFormat(BMDPixelFormat pxF);

    BMDPixelFormat GetPixelFormat() { return pixelFormat; }
    IDeckLink* _GetPortAddr() { return port; }
    IDeckLinkDisplayMode* _GetDisplayModeAddr() { return displayMode; }
    BMDDisplayMode GetDisplayMode() { return displayModes[selectedMode]->GetDisplayMode(); }
};

class DeckLinkOutputPort : public IDeckLinkPort
{
private:
    IDeckLinkOutput* output;
    IDeckLinkMutableVideoFrame* frame; // Mutable object ???
    DeckLinkPlaybackCallback* cb;

public:
    void enableVideo() override;
    void configure() override;
    bool doesSupportVideoMode() override;

    DeckLinkOutputPort(DeckLinkCard* card, IDeckLink* p, int mode = 0);
    ~DeckLinkOutputPort();
    void AddFrame(void* frameBuffer, size_t size = 0);
    void DisplayFrame();
};


class DeckLinkInputPort : public IDeckLinkPort
{
private:
    IDeckLinkInput* input;
    DeckLinkDevice* deckLinkCap;
    VideoFrameCallback* callback;

public:
    void enableVideo() override {}
    void configure() override {}
    bool doesSupportVideoMode() override { return true; }

    DeckLinkInputPort(DeckLinkCard* card, IDeckLink* p);
    ~DeckLinkInputPort();
    void RegisterVideoCallback(FrameArrivedCallback* _cb);
    void startCapture();
    DeckLinkDevice* _getPort() { return this->deckLinkCap; }
    std::queue<IDeckLinkVideoInputFrame*>* getQRef() { return callback->getQueRef(); }

};

class DeckLinkPlaybackCallback : public IDeckLinkVideoOutputCallback, public DeckLinkObject {
private:
    std::queue<IDeckLinkVideoFrame*> frames_q;
    IDeckLinkOutput* m_port;
    int count;
    BMDTimeValue timeValue, f_duration;
    BMDTimeScale scale;
    long long frames_count;
    std::mutex m_mutex;
public:
    DeckLinkPlaybackCallback(IDeckLinkOutput* dev);
    HRESULT ScheduledFrameCompleted(IDeckLinkVideoFrame* completedFrame, BMDOutputFrameCompletionResult result) override;
    void addFrame(IDeckLinkVideoFrame* frame);
    HRESULT ScheduledPlaybackHasStopped(void) override;

    // IUnknown interface
    HRESULT QueryInterface(REFIID iid, LPVOID* ppv);
    ULONG AddRef();
    ULONG Release();
};


class DeckLinkCard {
private:
    IDeckLinkIterator* iterator;
    std::unordered_map<int, DeckLinkOutputPort*> ports;
    std::unordered_map<int, DeckLinkInputPort*> inputPorts;
    std::vector<IDeckLink*> unconfiguredPorts;
    IDeckLink* port;
    HRESULT result;
    std::vector<int> selectedPorts;
    bool _selectedPort(int c);

public:
    DeckLinkCard();
    HRESULT checkError(bool fatal = false);
    DeckLinkOutputPort* SelectOutputPort(int idx, int mode = 1);
    DeckLinkInputPort* SelectInputPort(int c);
    ~DeckLinkCard();

};
