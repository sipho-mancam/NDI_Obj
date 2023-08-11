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


public:
    DeckLinkPort(IDeckLink* dev, bool o = true) :
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

        selectedMode = 40; //9; // 1080p50 1920 x 1080 50 fps 
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

public:
    // This is called on a seperate thread ...
    void arrived(IDeckLinkVideoInputFrame * frame) override {
        //std::cout << frame->GetWidth() << std::endl;
        frames_queue.push(frame);
    }

    // queue management 
    void clearAll()
    {
        while (!frames_queue.empty())
            frames_queue.pop();
    }
    IDeckLinkVideoInputFrame* getFrame()
    {
        IDeckLinkVideoInputFrame* temp = frames_queue.front();
        frames_queue.pop();
        return temp;
    }

    IDeckLinkVideoInputFrame* getFrameNoPop()
    {
        return frames_queue.front();
    }

    size_t queueSize() const { return frames_queue.size(); }
    void popTop() { frames_queue.pop(); }
    bool empty() const { return frames_queue.empty(); }



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
    

    void startCapture()
    {
        assert(deckLinkCap->startCapture(displayModes[selectedMode]->GetDisplayMode(), nullptr, true));
    }

    DeckLinkDevice* _getPort() { return this->deckLinkCap; }


};

class DeckLinkCard {
private:
    IDeckLinkIterator* iterator;
    std::vector<DeckLinkPort*> ports;
    std::vector<DeckLinkInputPort*> inputPorts;
    IDeckLink* port;
    HRESULT result;
    int sPort, selectedInport;
    DeckLinkPort* selectedOutputPort;

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
        bool flip =  true;
        int count = 0;
        while (iterator->Next(&port) == S_OK)
        {
            if (flip) {
                ports.push_back(new DeckLinkPort(port));
                count++;
                if (count >= 2) flip = false;
            }else{
                inputPorts.push_back(new DeckLinkInputPort(port));
            }
        }



        sPort = ports.size() - 1;

        selectedOutputPort = ports[sPort];

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
    DeckLinkPort* SelectPort(int idx)
    {
        if (idx >= 0 && idx < ports.size()) // Ports start counting from 1
        {
            sPort = idx;
            selectedOutputPort = ports[sPort];
            return selectedOutputPort;
        }
        return selectedOutputPort;
    }

    DeckLinkInputPort* SelectInputPort(int c)
    {
        if (c < 0 || c >= inputPorts.size())return nullptr;

        return inputPorts[c];
    }

    ~DeckLinkCard()
    {
        iterator->Release();

        for (auto& dev : this->ports)
        {
            delete dev;
        }

        CoUninitialize();
    }
};