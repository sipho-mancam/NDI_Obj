#include "decklink_api.hpp"


void VideoFrameObj::_updateRowBytes()
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

HRESULT VideoFrameObj::GetBytes(void** buffer) 
{
    if (data != nullptr)
    {
        *buffer = data;
        return S_OK;
    }
    *buffer = nullptr;
    return E_FAIL;
}

HRESULT VideoFrameObj::GetTimecode(BMDTimecodeFormat format, IDeckLinkTimecode** timecode)
{
    *timecode = NULL;
    return S_OK;
}

HRESULT VideoFrameObj::GetAncillaryData(IDeckLinkVideoFrameAncillary** anc_data)
{
    if (this->ancillaryData == nullptr) return E_FAIL;

    *anc_data = this->ancillaryData;
    return S_OK;
}

HRESULT VideoFrameObj::QueryInterface(REFIID id, void** outputInterface)
{
    return E_FAIL;
}

ULONG VideoFrameObj::AddRef()
{
    count += 1;
    return count;
}

ULONG VideoFrameObj::Release()
{
    count--;
    /*if (count == 0) {
        free(this->data);
    }*/
    return count;
}

void VideoFrameObj::SetPixelFormat(BMDPixelFormat pxF)
{
    this->pixelFormat = pxF;
    this->_updateRowBytes();
}

void VideoFrameObj::_testImageColorOut() // sets the image white  ;
{
    memset(this->data, 255, static_cast<size_t>(this->rowBytes) * this->height);
}

void VideoFrameObj::SetRowBytes(long bytes)
{
    this->rowBytes = bytes; 
    if (data)
        free(data);
    data = (void*)malloc(rowBytes * height);
}

void VideoFrameObj::SetFrameData(const void* fData, size_t s)
{
    if (s == 0 || s > static_cast<size_t>(this->rowBytes) * this->height)
    {
        memcpy(this->data, fData, (static_cast<size_t>(this->rowBytes) * this->height));
        return;
    }

    memcpy(this->data, fData, s);
}

VideoFrameObj::VideoFrameObj(long w, long h, BMDPixelFormat pxFormat, BMDFrameFlags flgs, void* d )
    : width(w), height(h), pixelFormat(pxFormat), flags(flgs), data(d)
{
    ancillaryData = nullptr;
    this->_updateRowBytes();
    if (data == nullptr) // allocate memory ...
    {
        data = malloc(static_cast<size_t>(this->rowBytes) * this->height);
    }
}

VideoFrameObj::~VideoFrameObj()
{
    this->Release();

    if (data)
        free(data);
}


MVideoObject::MVideoObject(long w, long h, BMDPixelFormat pxFormat, BMDFrameFlags flgs, void* d)
    : IDeckLinkMutableVideoFrame()
{
    
}



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
    : parent(par), 
    port(po), 
    displayMode(nullptr), 
    displayModeIterator(nullptr), 
    pixelFormat(bmdFormat8BitYUV),
    profileAttributes(nullptr), 
    selectedMode(0), 
    preview(false), 
    previewThread(nullptr), 
    running(false)
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
    : IDeckLinkPort(par, por), 
    cb(nullptr), 
    frame(nullptr), 
    frames_q(nullptr),
    rendering_thread(nullptr)
{
    // mode = 0 (HD) ... mode = 1 (UHD 4K)
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

void DeckLinkOutputPort::start()
{
    running = true;
    if (rendering_thread)
    {
        rendering_thread->join();
        delete rendering_thread;
    }
        
    rendering_thread = new std::thread(&DeckLinkOutputPort::run, this);
}

void DeckLinkOutputPort::stop()
{
    if (running)
    {
        running = false;
        if (rendering_thread)
        {
            rendering_thread->join();
            delete rendering_thread;
        }    
    }
}

void DeckLinkOutputPort::run()
{
    while (running)
    {
        if (frames_q != nullptr && !frames_q->empty())
        {
            VideoFrameObj* iframe = (VideoFrameObj*)frames_q->front();
            frames_q->pop();
            this->output->DisplayVideoFrameSync(iframe);
            stop_clock = std::chrono::high_resolution_clock::now();
            std::cout << ((stop_clock - start_clock).count() / 1000000) << " ms" << std::endl;
        }
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
    if (!frame)
    {
        IDeckLinkDisplayMode* d_mode = displayModes[selectedMode];
        result = output->CreateVideoFrame(
            d_mode->GetWidth(), 
            d_mode->GetHeight(), 
            d_mode->GetWidth()*2, 
            pixelFormat,
            bmdFrameFlagDefault, 
            &frame);

        if (result != S_OK)
        {
            frame = nullptr;
            return;
        }
        else {
            uchar* buffer;
            frame->GetBytes((void**) & buffer);
            memcpy(buffer, frameBuffer, size);
        }
        
    }
    else {
        uchar* buffer;
        frame->GetBytes((void**)&buffer);        
        memcpy(buffer, frameBuffer, size);
    }

    // if (pixelFormat == bmdFormat10BitYUV)
    //    std::cout << "I execute" << (pixelFormat == bmdFormat10BitYUV ? " 10bit YU" : "") << std::endl;

    if (frames_q == nullptr)
    {
        frames_q = new std::queue<IDeckLinkVideoFrame*>();
        frames_q->push(frame);
    }
    else {
       
        frames_q->push(frame);
    }

    if(!running)
       start();

    //this->output->DisplayVideoFrameSync(frame);

   /* BOOL playback_running;

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
    }*/
}

void DeckLinkOutputPort::DisplayFrame()
{
    this->output->DisplayVideoFrameSync(this->frame);
}

void DeckLinkOutputPort::subscribe_2_q(std::queue<IDeckLinkVideoFrame*>* q)
{
    if (q)
        frames_q = q;
}

DeckLinkOutputPort::~DeckLinkOutputPort()
{
   // if (port)port->Release();
    if (output)output->Release();
    if (displayModeIterator)displayModeIterator->Release();
    if (profileAttributes)profileAttributes->Release();
    if (preview && previewThread != nullptr)
    {
        previewThread->join();
        delete previewThread;
        preview = false;
    }

    this->stop(); // stop the rendering thread ...
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
        selectedMode = 0;
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
void DeckLinkInputPort::subscribe_2_input_q(std::queue<IDeckLinkVideoInputFrame*>* q)
{
    assert(q);
    callback->subscribe_2_q(q);
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
            //std::lock_guard<std::mutex> lock(m_mutex);
        }
        //std::cout << frames_q.size() << std::endl;
    }

    if (!frames_q.empty())
    {
        m_port->ScheduleVideoFrame(frames_q.front(), timeValue, f_duration, scale);
        frames_q.pop();
    }

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
    rgb_data_h(nullptr),
    frames_queue(nullptr)
{}


// This is called on a seperate thread ...
void VideoFrameCallback::arrived(IDeckLinkVideoInputFrame* frame) 
{
    start_clock = std::chrono::high_resolution_clock::now();
    frame->AddRef();
    // interogate the frame to decide how to process it ...
    width = frame->GetWidth();
    height = frame->GetHeight();

    switch (frame->GetPixelFormat())
    {
    case bmdFormat10BitYUV:
    {
        // from here we received the frame, now send it to the Interface_manager.
        if(frames_queue)
            frames_queue->push(frame); // pass it to the interface_manager

        //preview_10bit_yuv(frame);
        // set this to preview on a seperate thread ... 
        break;
    }
    case bmdFormat8BitYUV:
    {
        std::cout << "8-bit YUV received" << std::endl;
        break;
    }
    case bmdFormat10BitRGB:
    {
        std::cout << "10-bit RGB received" << std::endl;
        break;
    }

    case bmdFormat8BitBGRA:
    {
        std::cout << "8-bit BGRA received" << std::endl;
        break;
    }
    case bmdFormat12BitRGB:
    {
        std::cout << "12-bit RGB received" << std::endl;
        break;
    }
    }
}

void VideoFrameCallback::clearAll()
{
    while (!frames_queue->empty())
        frames_queue->pop();
}

void VideoFrameCallback::subscribe_2_q(std::queue<IDeckLinkVideoInputFrame*>* q)
{
    frames_queue = q;
}

void VideoFrameCallback::preview_10bit_yuv(IDeckLinkVideoInputFrame* frame)
{
    if (rgb_data_h == nullptr)
    {
        cudaMallocHost((void**)&pinnedMemory, frame->GetHeight() * frame->GetWidth() * sizeof(uint));
        cudaMalloc((void**)&gpuMemory, frame->GetRowBytes() * frame->GetHeight());

        cudaMalloc((void**)&rgb_data, frame->GetWidth() * frame->GetHeight() * sizeof(uchar3));
        cudaMallocHost((void**)&rgb_data_h, frame->GetWidth() * frame->GetHeight() * sizeof(uchar3));

        cudaMalloc((void**)&dst_4, frame->GetHeight() * (frame->GetWidth() / 2) * sizeof(uint4));
    }
    
    if (S_OK == frame->GetBytes((void**)&buffer))
    {
        cudaError_t cudaStatus = cudaMemcpy(gpuMemory, buffer, frame->GetRowBytes() * frame->GetHeight(), cudaMemcpyHostToDevice);

        assert(cudaStatus == cudaSuccess);
    }
    else {
        return;
    }

    this->unpack_10bit_yuv();
    convert_10bit_2_rgb();

    cv::Mat preview(cv::Size(width, height), CV_8UC3);
    // from here we build the NDI sender ....
    preview.data = (uchar*)rgb_data_h;
    cv::imshow("Preview", preview);
    cv::waitKey(2);

   /* cudaFree(gpuMemory);
    cudaFree(rgb_data);
    cudaFree(dst_4);

    cudaFreeHost(pinnedMemory);
    cudaFreeHost(rgb_data_h);*/
}

IDeckLinkVideoInputFrame* VideoFrameCallback::getFrame()
{
    if (frames_queue->empty()) return nullptr;
    IDeckLinkVideoInputFrame* temp = frames_queue->front();
    frames_queue->pop();
    return temp;
}

IDeckLinkVideoInputFrame* VideoFrameCallback::getFrameNoPop()
{
    if (frames_queue->empty()) return nullptr;
    return frames_queue->front();
}