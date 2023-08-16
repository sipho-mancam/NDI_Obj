#pragma once


#include <iostream>
#include <DecklinkAPI_h.h>
#include "platform.h"

#include <list>
#include <opencv2/opencv.hpp>
#include <vector>
#include <conio.h>
#include <cassert>
#include "DeckLinkDevice.h"
#include <queue>
#include <exception>
#include <unordered_map>
#include <mutex>

#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"

//#include "decklink_kernels.cuh"

#define CHECK_ERROR(result) \
                    if (result != S_OK)\
                    { \
                     std::cout << "There was an error"<< std::endl; \
                    }

// List of known pixel formats and their matching display names
static const std::list<std::pair<BMDPixelFormat, std::string>> gPixelFormats =
{
    { bmdFormat8BitYUV,     "8-bit YUV" },
    { bmdFormat10BitYUV,    "10-bit YUV" },
    { bmdFormat8BitARGB,    "8-bit ARGB" },
    { bmdFormat8BitBGRA,    "8-bit BGRA" },
    { bmdFormat10BitRGB,    "10-bit RGB" },
    { bmdFormat12BitRGB,    "12-bit RGB" },
    { bmdFormat12BitRGBLE,  "12-bit RGBLE" },
    { bmdFormat10BitRGBXLE, "10-bit RGBXLE" },
    { bmdFormat10BitRGBX,   "10-bit RGBX" },
};

static const std::list<std::pair<BMDVideoConnection, std::string>> gConnections =
{
    { bmdVideoConnectionUnspecified, "Unspecified Connection" },
    { bmdVideoConnectionSDI,         "SDI" },
    { bmdVideoConnectionHDMI,        "HDMI" },
    { bmdVideoConnectionOpticalSDI,  "Optical SDI" },
    { bmdVideoConnectionComponent,   "Component" },
    { bmdVideoConnectionComposite,   "Composite" },
    { bmdVideoConnectionSVideo,      "S-Video" },
};

static const std::list<std::pair<BMDSupportedVideoModeFlags, std::string>> gSDILinks =
{
    { bmdSupportedVideoModeSDISingleLink,	"Single-Link" },
    { bmdSupportedVideoModeSDIDualLink,		"Dual-Link" },
    { bmdSupportedVideoModeSDIQuadLink,		"Quad-Link" },
};




class VideoFrameObj : public IDeckLinkVideoFrame
{
private:
    long width, height, rowBytes;
    BMDPixelFormat pixelFormat;
    BMDFrameFlags flags;
    IDeckLinkVideoFrameAncillary* ancillaryData;
    void* data;
    ULONG count = 0;

    void _updateRowBytes()
    {
        long w = this->width;
        switch (this->pixelFormat)
        {
        case bmdFormat8BitYUV:
            rowBytes = (w * 16 / 8);
            break;
        case bmdFormat10BitYUV:
            rowBytes = (long)((w + 47.0) / 48.0) * 128;
            break;
        case bmdFormat8BitARGB:
            rowBytes = (w * 32 / 8);
            break;
        case bmdFormat8BitBGRA:
            rowBytes = (w * 32 / 8);
            break;
        case bmdFormat10BitRGB:
            rowBytes = (long)((w * 63.0) / 64.0) * 256;
            break;
        case bmdFormat12BitRGB:
            rowBytes = (long)((w * 36.0) / 8);
            break;
        case bmdFormat12BitRGBLE:
            rowBytes = (long)((w * 36 / 8.0));
            break;
        case bmdFormat10BitRGBXLE:
            rowBytes = (long)((w + 63.0) / 64.0) * 256;
            break;
        case bmdFormat10BitRGBX:
            rowBytes = (long)((w + 63.0) / 64.0) * 256;
            break;
        default:
            rowBytes = (long)((w * 3)); // assume the frame is a 3 channel 8-bit data channel.
            break;
        }

        if (data)
            free(data);

        data = malloc(static_cast<size_t>(this->rowBytes) * this->height);
    }

public:
    long GetWidth() override { return this->width; }
    long GetHeight() override { return this->height; }
    long GetRowBytes() override { return this->rowBytes; }
    BMDPixelFormat GetPixelFormat() override { return this->pixelFormat; }
    BMDFrameFlags GetFlags() override { return this->flags; }
    HRESULT GetBytes(void** buffer) override
    {
        if (data != nullptr)
        {
            *buffer = data;
            return S_OK;
        }
        *buffer = nullptr;
        return E_FAIL;
    }
    HRESULT GetTimecode(BMDTimecodeFormat format, IDeckLinkTimecode** timecode) override
    {
        *timecode = NULL;
        return S_OK;
    }
    HRESULT GetAncillaryData(IDeckLinkVideoFrameAncillary** anc_data) override
    {
        if (this->ancillaryData == nullptr) return E_FAIL;

        *anc_data = this->ancillaryData;
        return S_OK;
    }

    HRESULT QueryInterface(REFIID id, void** outputInterface)
    {
        return E_FAIL;
    }

    ULONG AddRef()
    {
        count += 1;
        return count;
    }

    ULONG Release()
    {
        count--;
        /*if (count == 0) {
            free(this->data);
        }*/
        return count;
    }

    void SetRowBytes(long bytes) { this->rowBytes = bytes; }
    void SetWidth(long w) { this->width = w; this->_updateRowBytes(); }
    void SetHeight(long h) { this->height = h; this->_updateRowBytes(); }
    void SetPixelFormat(BMDPixelFormat pxF)
    {
        this->pixelFormat = pxF;
        this->_updateRowBytes();
    }
    void SetFrameFlags(BMDFrameFlags f) { this->flags = f; }

    void _testImageColorOut() // sets the image white  ;
    {
        memset(this->data, 255, static_cast<size_t>(this->rowBytes) * this->height);
    }

    void SetFrameData(const void* fData, size_t s)
    {
        if (s == 0 || s > static_cast<size_t>(this->rowBytes) * this->height)
        {
            memcpy(this->data, fData, (static_cast<size_t>(this->rowBytes) * this->height));
            return;
        }

        memcpy(this->data, fData, s);

    }

    VideoFrameObj(long w, long h, BMDPixelFormat pxFormat, BMDFrameFlags flgs = bmdFrameFlagDefault, void* d = nullptr)
        : width(w), height(h), pixelFormat(pxFormat), flags(flgs), data(d)
    {
        ancillaryData = nullptr;
        this->_updateRowBytes();
        if (data == nullptr) // allocate memory ...
        {
            data = malloc(static_cast<size_t>(this->rowBytes) * this->height);
        }
    }

    ~VideoFrameObj()
    {
        this->Release();
    }

};

class DeckLinkPlaybackCallback : public IDeckLinkVideoOutputCallback {
private:
    std::queue<IDeckLinkVideoFrame*> frames_q;
    HRESULT result;
    IDeckLinkOutput* m_port;
    int count;
    BMDTimeValue timeValue, f_duration;
    BMDTimeScale scale;
    long long frames_count;
    std::mutex m_mutex;
public:

    DeckLinkPlaybackCallback(IDeckLinkOutput* dev)
        : m_port(dev),
        result(S_OK), count(0), scale(50000), f_duration(1000), frames_count(0), timeValue(0)
    {}
    HRESULT ScheduledFrameCompleted(IDeckLinkVideoFrame* completedFrame, BMDOutputFrameCompletionResult result) override
    {
        timeValue += f_duration;
        BMDTimeValue frameCompletionTimestamp;

        if (completedFrame)
        {
            if (m_port->GetFrameCompletionReferenceTimestamp(completedFrame, scale, &frameCompletionTimestamp) == S_OK)
            {
                std::lock_guard<std::mutex> lock(m_mutex);

            }
        }

        if (!frames_q.empty())
        {
            m_port->ScheduleVideoFrame(frames_q.front(), timeValue, f_duration, scale);
            frames_q.pop();
        }
        std::cout << frames_q.size() << std::endl;

        return S_OK;
    }

    void addFrame(IDeckLinkVideoFrame* frame)
    {
        frames_q.push(frame);
    }

    HRESULT ScheduledPlaybackHasStopped(void) override
    {
        IDeckLinkVideoFrame* vid_frame;
        while (frames_q.empty())
        {
            vid_frame = frames_q.front();
            frames_q.pop();
            vid_frame->Release();
        }

        frames_count = 0;

        return S_OK;
    }

    // IUnknown interface
    HRESULT QueryInterface(REFIID iid, LPVOID* ppv)
    {
        HRESULT result = S_OK;

        if (ppv == nullptr)
            return E_INVALIDARG;

        // Obtain the IUnknown interface and compare it the provided REFIID
        if (iid == IID_IUnknown)
        {
            *ppv = this;
            AddRef();
        }
        else if (iid == IID_IDeckLinkVideoOutputCallback)
        {
            *ppv = (IDeckLinkVideoOutputCallback*)this;
            AddRef();
        }
        else if (iid == IID_IDeckLinkAudioOutputCallback)
        {
            *ppv = (IDeckLinkAudioOutputCallback*)this;
            AddRef();
        }
        else
        {
            *ppv = nullptr;
            result = E_NOINTERFACE;
        }

        return result;
    }

    ULONG AddRef()
    {
        count += 1;
        return count;
    }

    ULONG Release()
    {
        count--;
        ULONG newRefValue = --count;

        if (newRefValue == 0)
            delete this;

        return newRefValue;
    }


};

class DeckLinkPort {
protected:
    IDeckLink* port;
    bool isOutput;
    IDeckLinkOutput* output;
    IDeckLinkInput* input;
    HRESULT result;
    IDeckLinkDisplayModeIterator* displayModeIterator;
    IDeckLinkDisplayMode* displayMode;
    std::vector<IDeckLinkDisplayMode*> displayModes;

    IDeckLinkProfileAttributes* profileAttrib;

    BMDPixelFormat pixelFormat;
    int displayModesCount;
    unsigned int selectedMode;

    VideoFrameObj* frame;

    DeckLinkPlaybackCallback* cb;


public:
    // mode = 0 (HD) ... mode = 1 (UHD 4K)
    DeckLinkPort(IDeckLink* dev, bool o = true, int mode = 0) :
        port(dev), isOutput(o), result(S_OK), displayMode(nullptr), displayModeIterator(nullptr), pixelFormat(bmdFormat10BitYUV),
        profileAttrib(nullptr), displayModesCount(0)
    {
       
        output = nullptr;
        assert(dev != nullptr);
        if (isOutput)
        {
            result = port->QueryInterface(IID_IDeckLinkOutput, (void**)&this->output);
            checkError("Creating Output Device pointer...");

            if (result == S_OK)
            {
                cb = new DeckLinkPlaybackCallback(this->output);

                result = this->output->GetDisplayModeIterator(&displayModeIterator);

                assert(result == S_OK);

                while (displayModeIterator->Next(&displayMode) == S_OK)
                {
                    displayModesCount++;
                    displayModes.push_back(displayMode);
                }

                result = this->output->QueryInterface(IID_IDeckLinkProfileAttributes, (void**)&profileAttrib);
                assert(result == S_OK);
            }

        }
        else {
            result = port->QueryInterface(IID_IDeckLinkOutput, (void**)&this->input);
            checkError("Creating Output Device pointer...");

            if (result == S_OK)
            {
                result = this->input->GetDisplayModeIterator(&displayModeIterator);

                assert(result == S_OK);

                while (displayModeIterator->Next(&displayMode) == S_OK)
                {
                    displayModesCount++;
                    displayModes.push_back(displayMode);
                }

                result = this->input->QueryInterface(IID_IDeckLinkProfileAttributes, (void**)&profileAttrib);
                assert(result == S_OK);
            }
        }

        selectedMode = mode == 1? 45 : 9; //9; // 1080p50 1920 x 1080 50 fps 
        displayMode = displayModes[selectedMode];

        frame = new VideoFrameObj(displayMode->GetWidth(), displayMode->GetHeight(), pixelFormat);

        configure();
    }

    bool isOutputPort() { return this->isOutput; }

    BMDPixelFormat GetPixelFormat() { return pixelFormat; }
    IDeckLink* _GetPortAddr() { return port; }
    IDeckLinkOutput* _GetOutputAddr() { return output; }
    IDeckLinkDisplayMode* _GetDisplayModeAddr() { return displayMode; }
    VideoFrameObj* GetFrameObject() { return this->frame; }

    BMDDisplayMode GetDisplayMode() {
        return displayModes[selectedMode]->GetDisplayMode();
    }

    void SetDisplayMode(int c)
    {
        if (c < 0 || c >= displayModes.size()) return;

        selectedMode = c;
        displayMode = displayModes[selectedMode];
    }

    void SetPixelFormat(BMDPixelFormat pxF)
    {
        pixelFormat = pxF;
        frame->SetPixelFormat(pxF);
    }

    void SetRowBytes(long bytes) { frame->SetRowBytes(bytes); }

    bool doesSupportVideoMode()
    {
        dlbool_t displayModeSupported;

        result = output->DoesSupportVideoMode(
            bmdVideoConnectionUnspecified,
            displayModes[selectedMode]->GetDisplayMode(),
            pixelFormat,
            bmdNoVideoOutputConversion,
            bmdSupportedVideoModeDefault,
            NULL,
            &displayModeSupported
        );

        assert(result == S_OK);

        return displayModeSupported == true;
    }

    void enableVideo()
    {
        result = output->EnableVideoOutput(displayModes[selectedMode]->GetDisplayMode(), bmdVideoOutputFlagDefault);
        assert(result == S_OK);
    }

    void configure()
    {
        if (isOutput)
        {
            assert(doesSupportVideoMode());
            enableVideo();
        }
    }


    void AddFrame(void* frameBuffer, size_t size = 0)
    {
        frame->SetFrameData(frameBuffer, size);
        BOOL playback_running;
       
        this->output->IsScheduledPlaybackRunning(&playback_running);

        if (!playback_running)
        {
            BMDTimeValue tv = 0, duration = 1000;
            BMDTimeScale scale = 50000;

            displayMode->GetFrameRate(&duration, &scale);

            this->output->SetScheduledFrameCompletionCallback(cb);

            this->output->ScheduleVideoFrame(frame, tv, duration, scale);

            this->output->StartScheduledPlayback(0, scale, 1);
        }
        else {
            cb->addFrame(frame);
        }
    }

    void DisplayFrame()
    {
        this->output->DisplayVideoFrameSync(this->frame);
    }


    HRESULT checkError(std::string info = "", bool fatal = false)
    {
        if (result != S_OK)
        {
            std::cerr << "Decklink API faild: " << info << std::endl;
            if (fatal) return (HRESULT)-1L;
            return result;
        }
        return result;
    }

    ~DeckLinkPort()
    {

        port->Release();
        output->Release();
        displayModeIterator->Release();
        profileAttrib->Release();
    }

};


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
    
    uchar3* rgb_data, *rgb_data_h;

    std::mutex mtx;

public:

    VideoFrameCallback(int mFrameCount = 5) : 
        maxFrameCount(mFrameCount) ,
        droppedFrames(false),
        init(false),
        pinnedMemory(nullptr),
        gpuMemory(nullptr),
        height(0), width(0),
        dst_4(nullptr), dst_full(nullptr), buffer(NULL), 
        pxFormat(bmdFormatUnspecified),
        rgb_data(nullptr), rgb_data_h(nullptr)
    {
    
    }
    // This is called on a seperate thread ...
    void arrived(IDeckLinkVideoInputFrame * frame) override {
        
       frames_queue.push(frame);      
        //std::lock_guard<std::mutex> lock(mtx);

        if (pxFormat != frame->GetPixelFormat())
        {
            pxFormat = frame->GetPixelFormat();

            if (pinnedMemory != nullptr)
            {
                cudaFreeHost(pinnedMemory);
                cudaFree(gpuMemory);

                if (dst_4 != nullptr)
                {
                    cudaFree(dst_4);
                    cudaFree(dst_full);
                }
            }

            assert(cudaSuccess==cudaMallocHost((void**)&pinnedMemory, frame->GetHeight() * frame->GetWidth() * sizeof(uint)));
            assert(cudaSuccess==cudaMalloc((void**)&gpuMemory, frame->GetRowBytes() * frame->GetHeight()));

            assert(cudaSuccess == cudaMalloc((void**)&rgb_data, frame->GetWidth() * frame->GetHeight() * sizeof(uchar3)));
            assert(cudaSuccess == cudaMallocHost((void**)&rgb_data_h, frame->GetWidth() * frame->GetHeight() * sizeof(uchar3)));
            // this assumes we are receiving YUV data at 10bits.
            switch (frame->GetPixelFormat())
            {
            case bmdFormat10BitYUV:
            {
                assert(cudaSuccess == cudaMalloc((void**)&dst_4, frame->GetHeight() * (frame->GetWidth() / 2) * sizeof(uint4)));
                assert(cudaSuccess == cudaMalloc((void**)&dst_full, frame->GetHeight() * frame->GetWidth() * sizeof(uint)));
                break;
            }
            case bmdFormat8BitYUV:
            {
                assert(cudaSuccess == cudaMalloc((void**)&dst_4, frame->GetHeight() * (frame->GetWidth() / 2) * sizeof(uint4)));
                assert(cudaSuccess == cudaMalloc((void**)&dst_full, frame->GetHeight() * frame->GetWidth() * sizeof(uint)));
                break;
            }
            }  
        }
      
        width = frame->GetWidth();
        height = frame->GetHeight();

        if (S_OK == frame->GetBytes((void**)&buffer)) 
        {
            cudaError_t cudaStatus = cudaMemcpy(gpuMemory, buffer, frame->GetRowBytes() * frame->GetHeight(), cudaMemcpyHostToDevice);
            
            assert(cudaStatus == cudaSuccess);
            
            switch (frame->GetPixelFormat())
            {
            case bmdFormat10BitYUV:
            {
                //std::cout << "I received data" << std::endl;
                this->unpack_10bit_yuv();
                convert_10bit_2_rgb();
                cv::namedWindow("Preview", cv::WINDOW_NORMAL);
                cv::Mat preview(cv::Size(width, height), CV_8UC3);

                    
                // from here we build the NDI sender ....
                preview.data = (uchar*)rgb_data_h;

                cv::imshow("Preview", preview);
                cv::waitKey(2);
                break;
            }
          
            }
        }
       

    
    }

    std::queue<IDeckLinkVideoInputFrame*>* getQueRef() { return(&frames_queue); }

    void convert_10bit_2_rgb();

    void unpack_10bit_yuv();


    // queue management 
    void clearAll()
    {
        while (!frames_queue.empty())
            frames_queue.pop();
    }
    IDeckLinkVideoInputFrame* getFrame()
    {
        if (frames_queue.empty()) return nullptr;
        IDeckLinkVideoInputFrame* temp = frames_queue.front();
        frames_queue.pop();
        return temp;
    }

    IDeckLinkVideoInputFrame* getFrameNoPop()
    {
        if (frames_queue.empty()) return nullptr;
        return frames_queue.front();
    }

    size_t queueSize() const { return frames_queue.size(); }
    void popTop() { frames_queue.pop(); }
    bool empty() const { return frames_queue.empty(); }
    bool frameDropped() { return droppedFrames; }
    bool overflow() { return frames_queue.size() == maxFrameCount; }
};

class DeckLinkInputPort : public DeckLinkPort {
private:
   
    DeckLinkDevice* deckLinkCap;
    VideoFrameCallback* callback;


public:
    DeckLinkInputPort(IDeckLink* dev) : DeckLinkPort(dev, false)
    {
        callback = new VideoFrameCallback();
        deckLinkCap = new DeckLinkDevice(dev);

        deckLinkCap->init();

        deckLinkCap->registerFrameArrivedCallback(callback);
    }
    ~DeckLinkInputPort()
    {
        deckLinkCap->stopCapture();
        delete deckLinkCap;
    }

    void RegisterVideoCallback(FrameArrivedCallback* _cb) {
        deckLinkCap->registerFrameArrivedCallback(_cb);
        delete callback;
        // might cause bugs when the callback object is different from the one we created here ...
        callback = (VideoFrameCallback*)_cb;
    }
    

    void startCapture()
    {
        assert(deckLinkCap->startCapture(displayModes[selectedMode]->GetDisplayMode(), nullptr, true));
    }

    DeckLinkDevice* _getPort() { return this->deckLinkCap; }

    std::queue< IDeckLinkVideoInputFrame*>* getQRef() { return callback->getQueRef(); }


};

class DeckLinkCard {
private:
    IDeckLinkIterator* iterator;
    std::unordered_map<int, DeckLinkPort*> ports;
    std::unordered_map<int, DeckLinkInputPort*> inputPorts;

    std::vector<IDeckLink*> unconfiguredPorts;

    IDeckLink* port;
    HRESULT result;
    int sPort, selectedInport;
    DeckLinkPort* selectedOutputPort;

    std::vector<int> selectedPorts;

    bool _selectedPort(int c)
    {
        for (int elem : selectedPorts)
        {
            if (elem == c) return true;
        }
        return false;
    }

public:
    DeckLinkCard()
    {
        selectedOutputPort = nullptr;

        result = CoInitializeEx(NULL, COINITBASE_MULTITHREADED);
        checkError();
        result = GetDeckLinkIterator(&iterator);
        checkError(true);
        selectedInport = 0;

        if (result != S_OK) return;

        while (iterator->Next(&port) == S_OK)
        {
            unconfiguredPorts.push_back(port);
        }

        std::cout << "Decklink Device Initialized successfully ..." << std::endl;
    }

    HRESULT checkError(bool fatal = false)
    {
        if (result != S_OK)
        {
            std::cerr << "Decklink API faild: " << result << std::endl;
            if (fatal) return -1;
            return result;
        }
        return result;
    }

    DeckLinkPort* GetCurrentPort() { return this->selectedOutputPort; }
    DeckLinkPort* SelectPort(int idx, int mode = 1)
    {
        if (idx >= 0 && idx < unconfiguredPorts.size()) // Ports start counting from 1
        {
            if (!_selectedPort(idx))
            {
                DeckLinkPort* p = new DeckLinkPort(unconfiguredPorts[idx], true, mode);
                ports[idx] = p;
                selectedPorts.push_back(idx);
                return p;
            }
            else {
                // if it is selected, but it was already created ... just return it.
                try {
                    return ports.at(idx);
                }
                catch (std::out_of_range& e) {
                    std::cerr << "Port Already selected as input port..." << std::endl;
                    return nullptr;
                }
            }
            
        }
        return nullptr;
    }

    DeckLinkInputPort* SelectInputPort(int c)
    {
        if (c < 0 || c >= unconfiguredPorts.size())return nullptr;

        if (!_selectedPort(c)) {
            DeckLinkInputPort* p = new DeckLinkInputPort(unconfiguredPorts[c]);
            inputPorts[c] = p;
            selectedPorts.push_back(c);
            return p;
        }
        else {
            try {
                return inputPorts.at(c);
            }catch (std::out_of_range& or ) {
                std::cerr << "Port already selected as output port" << std::endl;
                return nullptr;
            }
        }

        return inputPorts[c];
    }

    ~DeckLinkCard()
    {
        iterator->Release();

        for (auto& dev : this->ports)
        {
            delete dev.second;
        }

        for (auto& dev : this->inputPorts)
        {
            delete dev.second;
        }

        this->ports.clear();
        this->inputPorts.clear();

        CoUninitialize();
    }
};