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


#define PREROLL 3
#ifndef assert
#define assert(e) ()
#endif // !assert

#define HD_MODE 0
#define UHD_MODE 1


class DeckLinkCard;
class DeckLinkPlaybackCallback;
static std::chrono::steady_clock::time_point start_clock, stop_clock;

class VideoFrameObj : public IDeckLinkVideoFrame
{
private:
    long width, height, rowBytes;
    BMDPixelFormat pixelFormat;
    BMDFrameFlags flags;
    IDeckLinkVideoFrameAncillary* ancillaryData;
    void* data;
    ULONG count = 0;

    void _updateRowBytes();

public:
    long GetWidth() override { return this->width; }
    long GetHeight() override { return this->height; }
    long GetRowBytes() override { return this->rowBytes; }
    BMDPixelFormat GetPixelFormat() override { return this->pixelFormat; }
    BMDFrameFlags GetFlags() override { return this->flags; }
    void SetRowBytes(long bytes); 
    void SetWidth(long w) { this->width = w; this->_updateRowBytes(); }
    void SetHeight(long h) { this->height = h; this->_updateRowBytes(); }
    void SetFrameFlags(BMDFrameFlags f) { this->flags = f; }

    HRESULT GetBytes(void** buffer) override;
    HRESULT GetTimecode(BMDTimecodeFormat format, IDeckLinkTimecode** timecode) override;
    HRESULT GetAncillaryData(IDeckLinkVideoFrameAncillary** anc_data) override;
    HRESULT QueryInterface(REFIID id, void** outputInterface) override;
    ULONG AddRef() override;
    ULONG Release() override;
    void SetPixelFormat(BMDPixelFormat pxF);
    void _testImageColorOut(); // sets the image white  ;
    void SetFrameData(const void* fData, size_t s=0);
    VideoFrameObj(long w, long h, BMDPixelFormat pxFormat, BMDFrameFlags flgs = bmdFrameFlagDefault, void* d = nullptr);
    ~VideoFrameObj();
};

class MVideoObject : public IDeckLinkMutableVideoFrame // experimental ...
{
public:
    MVideoObject(long w, long h, BMDPixelFormat pxFormat, BMDFrameFlags flgs = bmdFrameFlagDefault, void* d = nullptr);
};

class VideoFrameCallback : public FrameArrivedCallback {
private:
    std::queue<IDeckLinkVideoInputFrame*>* frames_queue;
    std::queue<IDeckLinkVideoFrame*>* frames_q;
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
    void preview_10bit_yuv(IDeckLinkVideoInputFrame* frame);
    std::queue<IDeckLinkVideoInputFrame*>* getQueRef() { return(frames_queue); }
    void subscribe_2_q(std::queue<IDeckLinkVideoInputFrame*>* q);
    void subscribe_2_q(std::queue<IDeckLinkVideoFrame*>* q);
    void convert_10bit_2_rgb(); //cuda_function
    void unpack_10bit_yuv(); // cuda_function 

    // queue management 
    void clearAll();
    IDeckLinkVideoInputFrame* getFrame();
    IDeckLinkVideoInputFrame* getFrameNoPop();

    size_t queueSize() const { return frames_queue->size(); }
    void popTop() { frames_queue->pop(); }
    bool empty() const { return frames_queue->empty(); }
    bool frameDropped() { return droppedFrames; }
    bool overflow() { return frames_queue->size() == maxFrameCount; }
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

    std::thread* previewThread;

    bool preview, running;

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
    void enablePreview() { preview = true; }
    void disablePreview() { preview = false; }
};

class DeckLinkOutputPort : public IDeckLinkPort
{
protected:
    IDeckLinkOutput* output;
    IDeckLinkMutableVideoFrame* frame; 
    IDeckLinkMutableVideoFrame* srcFrame;
    IDeckLinkVideoConversion* conversion;
    DeckLinkPlaybackCallback* cb;
    std::queue<IDeckLinkVideoFrame*>* frames_q;
    std::thread* rendering_thread;

    bool m_referenceLocked;

    bool* _release_frames; // this is the flag used to synchronize between multiple outputs
    virtual void run();

public:
    void enableVideo() override;
    void configure() override;
    bool doesSupportVideoMode() override;

    DeckLinkOutputPort(DeckLinkCard* card, IDeckLink* p, int mode = 0);
    ~DeckLinkOutputPort();

    void AddFrame(void* frameBuffer, size_t size = 0);
    void DisplayFrame();
    void setPixelFormat(BMDPixelFormat f) { pixelFormat = f; }
    void playFrameBack(); // play back the frame asynchronously.

    void synchronize(bool* _sync_flag);
    void subscribe_2_q(std::queue<IDeckLinkVideoFrame*>* q); // this q gives us data to output ...
    std::queue<IDeckLinkVideoFrame*>* get_output_q();

    bool waitForReference(); // waits for as long as the reference signal is not obtained.
    
    void start();
    void stop();
};

class CameraOutputPort : public DeckLinkOutputPort
{
private:
    void run() override;

public:
    CameraOutputPort(DeckLinkCard* card, IDeckLink* p, int mode = 0);
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

    void subscribe_2_input_q(std::queue<IDeckLinkVideoInputFrame*>* q);
    void subscribe_2_input_q(std::queue<IDeckLinkVideoFrame*>* q);
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

    BMDTimeValue getCurrentDisplayTime() { return timeValue;  }
    BMDTimeValue getFrameDuration() { return f_duration; }
    BMDTimeScale getTimeScale() { return scale; }

    void setDuration(BMDTimeValue d) { f_duration = d; }
    void setTimeScale(BMDTimeScale s) { scale = s; }
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
    CameraOutputPort* SelectCamOutputPort(int idx, int mode = 1);
    DeckLinkInputPort* SelectInputPort(int c);
    ~DeckLinkCard();

};
