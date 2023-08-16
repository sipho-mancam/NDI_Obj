#include "decklink_api.hpp"


DeckLinkCard::DeckLinkCard()
{
    result = CoInitializeEx(NULL, COINITBASE_MULTITHREADED);
    checkError();
    result = GetDeckLinkIterator(&iterator);
    checkError(true);

    if (result != S_OK) return;

    while (iterator->Next(&port) == S_OK)
    {
        unconfiguredPorts.push_back(port);
    }
    std::cout << "Decklink Device Initialized successfully ..." << std::endl;
}

HRESULT DeckLinkCard::checkError(bool fatal)
{
    if (result != S_OK)
    {
        std::cerr << "Decklink API faild: " << result << std::endl;
        if (fatal) return -1;
        return result;
    }
    return result;
}


DeckLinkOutputPort* DeckLinkCard::SelectOutputPort(int idx, int mode)
{
    if (idx >= 0 && idx < unconfiguredPorts.size()) // Ports start counting from 1
    {
        if (!_selectedPort(idx))
        {
            DeckLinkOutputPort* p = new DeckLinkOutputPort(this, unconfiguredPorts[idx], mode);
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

DeckLinkInputPort* DeckLinkCard::SelectInputPort(int c)
{
    if (c < 0 || c >= unconfiguredPorts.size())return nullptr;

    if (!_selectedPort(c)) {
        DeckLinkInputPort* p = new DeckLinkInputPort(this, unconfiguredPorts[c]);
        inputPorts[c] = p;
        selectedPorts.push_back(c);
        return p;
    }
    else {
        try {
            return inputPorts.at(c);
        }
        catch (std::out_of_range& or ) {
            std::cerr << "Port already selected as output port" << std::endl;
            return nullptr;
        }
    }

    return inputPorts[c];
}

DeckLinkCard::~DeckLinkCard()
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

bool DeckLinkCard::_selectedPort(int c)
{
    for (int elem : selectedPorts)
    {
        if (elem == c) return true;
    }
    return false;
}


HRESULT DeckLinkObject::checkError(std::string info , bool fatal)
{
    if (result != S_OK)
    {
        std::cerr << std::string("Decklink API faild: ")+ std::string(__FILE__) << info << std::endl;
        if (fatal) return (HRESULT)-1L;
        return result;
    }
    return result;
}


IDeckLinkPort::IDeckLinkPort(DeckLinkCard* par, IDeckLink* po)
    : parent(par), port(po), displayMode(nullptr), displayModeIterator(nullptr), pixelFormat(bmdFormat10BitYUV),
    profileAttributes(nullptr)
{
}

void IDeckLinkPort::SetDisplayMode(int c)
{
    if (c < 0 || c >= displayModes.size()) return;

    selectedMode = c;
    displayMode = displayModes[selectedMode];
}

void IDeckLinkPort::SetPixelFormat(BMDPixelFormat pxF)
{
    pixelFormat = pxF;
}

DeckLinkOutputPort::DeckLinkOutputPort(DeckLinkCard* par, IDeckLink* por, int mode)
    : IDeckLinkPort(par, por)
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
            displayModes.push_back(displayMode);
        }
        result = this->output->QueryInterface(IID_IDeckLinkProfileAttributes, (void**)&profileAttributes);
        assert(result == S_OK);

        selectedMode = mode == 1 ? 45 : 9; //9; // 1080p50 1920 x 1080 50 fps 
        displayMode = displayModes[selectedMode];

        // create mutable videoframe object ...
        // TODO: here..

        // configure the video out ...
        configure();
    }
}

bool DeckLinkOutputPort::doesSupportVideoMode()
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

void DeckLinkOutputPort::enableVideo()
{
    result = output->EnableVideoOutput(displayModes[selectedMode]->GetDisplayMode(), bmdVideoOutputFlagDefault);
    assert(result == S_OK);
}

void DeckLinkOutputPort::configure()
{   
    assert(doesSupportVideoMode());
    enableVideo();
}

void DeckLinkOutputPort::AddFrame(void* frameBuffer, size_t size)
{
    
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

void DeckLinkOutputPort::DisplayFrame()
{
    this->output->DisplayVideoFrameSync(this->frame);
}

DeckLinkOutputPort::~DeckLinkOutputPort()
{
    if (port)port->Release();
    if (output)output->Release();
    if (displayModeIterator)displayModeIterator->Release();
    if (profileAttributes)profileAttributes->Release();
}


DeckLinkInputPort::DeckLinkInputPort(DeckLinkCard* card, IDeckLink* p) : IDeckLinkPort(card, p)
{
    result = port->QueryInterface(IID_IDeckLinkOutput, (void**)&this->input);
    assert(result == S_OK);

    if (result == S_OK)
    {
        result = this->input->GetDisplayModeIterator(&displayModeIterator);
        assert(result == S_OK);

        while (displayModeIterator->Next(&displayMode) == S_OK)
        {
            displayModes.push_back(displayMode);
        }

        result = this->input->QueryInterface(IID_IDeckLinkProfileAttributes, (void**)&profileAttributes);
        assert(result == S_OK);


        callback = new VideoFrameCallback();
        deckLinkCap = new DeckLinkDevice(p);

        deckLinkCap->init();

        deckLinkCap->registerFrameArrivedCallback(callback);
    }
}

DeckLinkInputPort::~DeckLinkInputPort()
{
    deckLinkCap->stopCapture();
    delete deckLinkCap;
}


void DeckLinkInputPort::RegisterVideoCallback(FrameArrivedCallback* _cb)
{
    deckLinkCap->registerFrameArrivedCallback(_cb);
    delete callback;
    // might cause bugs when the callback object is different from the one we created here ...
    callback = (VideoFrameCallback*)_cb;
}

void DeckLinkInputPort::startCapture()
{
    assert(deckLinkCap->startCapture(displayModes[selectedMode]->GetDisplayMode(), nullptr, true));
}


DeckLinkPlaybackCallback::DeckLinkPlaybackCallback(IDeckLinkOutput* dev)
    : m_port(dev),
     count(0), scale(50000), f_duration(1000), frames_count(0), timeValue(0)
{}


HRESULT DeckLinkPlaybackCallback::ScheduledFrameCompleted(IDeckLinkVideoFrame* completedFrame, BMDOutputFrameCompletionResult result)
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

void DeckLinkPlaybackCallback::addFrame(IDeckLinkVideoFrame* frame)
{
    frames_q.push(frame);
}


HRESULT DeckLinkPlaybackCallback::ScheduledPlaybackHasStopped(void)
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


HRESULT DeckLinkPlaybackCallback::QueryInterface(REFIID iid, LPVOID* ppv)
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

ULONG DeckLinkPlaybackCallback::AddRef()
{
    count += 1;
    return count;
}

ULONG DeckLinkPlaybackCallback::Release()
{
    count--;
    ULONG newRefValue = --count;

    if (newRefValue == 0)
        delete this;

    return newRefValue;
}

VideoFrameCallback::VideoFrameCallback(int mFrameCount) :
    maxFrameCount(mFrameCount),
    droppedFrames(false),
    init(false),
    pinnedMemory(nullptr),
    gpuMemory(nullptr),
    height(0), width(0),
    dst_4(nullptr), dst_full(nullptr), buffer(NULL),
    pxFormat(bmdFormatUnspecified),
    rgb_data(nullptr),
    rgb_data_h(nullptr)
{}


// This is called on a seperate thread ...
void VideoFrameCallback::arrived(IDeckLinkVideoInputFrame* frame) {

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

        assert(cudaSuccess == cudaMallocHost((void**)&pinnedMemory, frame->GetHeight() * frame->GetWidth() * sizeof(uint)));
        assert(cudaSuccess == cudaMalloc((void**)&gpuMemory, frame->GetRowBytes() * frame->GetHeight()));

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

void VideoFrameCallback::clearAll()
{
    while (!frames_queue.empty())
        frames_queue.pop();
}

IDeckLinkVideoInputFrame* VideoFrameCallback::getFrame()
{
    if (frames_queue.empty()) return nullptr;
    IDeckLinkVideoInputFrame* temp = frames_queue.front();
    frames_queue.pop();
    return temp;
}

IDeckLinkVideoInputFrame* VideoFrameCallback::getFrameNoPop()
{
    if (frames_queue.empty()) return nullptr;
    return frames_queue.front();
}