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

VideoFrameObj::VideoFrameObj(long w, long h, BMDPixelFormat pxFormat, BMDFrameFlags flgs, void* d)
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

CameraOutputPort::CameraOutputPort(DeckLinkCard* card, IDeckLink* p, int mode)
    : DeckLinkOutputPort(card, p, mode)
{}

CameraOutputPort* DeckLinkCard::SelectCamOutputPort(int idx, int mode)
{
    if (idx >= 0 && idx < unconfiguredPorts.size()) // Ports start counting from 1
    {
        if (!_selectedPort(idx))
        {
            CameraOutputPort* p = new CameraOutputPort(this, unconfiguredPorts[idx], mode);
            ports[idx] = p;
            selectedPorts.push_back(idx);
            return p;
        }
        else {
            // if it is selected, but it was already created ... just return it.
            try {
                return (CameraOutputPort*)ports.at(idx);
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

HRESULT DeckLinkObject::checkError(std::string info, bool fatal)
{
    if (result != S_OK)
    {
        std::cerr << std::string("Decklink API faild: ") + std::string(__FILE__) << info << std::endl;
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
    pixelFormat(bmdFormat10BitYUV),
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
    rendering_thread(nullptr),
    srcFrame(nullptr),
    conversion(nullptr),
    _release_frames(nullptr),
    m_referenceLocked(false),
    selectedDMode(nullptr)
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

        selectedMode = getModeIndex(mode);
        displayMode = displayModes[selectedMode];
        width = displayMode->GetWidth();
        height = displayMode->GetHeight();

        configure();
    }
}

int DeckLinkOutputPort::getModeIndex(int mode)
{
    std::string hd_mode_s = "1080i50";
    std::string uhd_mode = "2160p50";

    int s_mode = -1;

    BSTR d_name;
    int j;
    // find the right string that point to 1080i or 2160p
    for (j = 0; j < displayModes.size(); j++)
    {
        displayModes[j]->GetName(&d_name);
        std::string comp = DlToStdString(d_name);
        switch (mode)
        {
        case HD_MODE:
        {
            if (hd_mode_s == comp)
                s_mode = j;
            break;
        }
        case UHD_MODE:
        {
            if (uhd_mode == comp)
                s_mode = j;
            break;
        }
        }

        if (s_mode != -1)
            return s_mode;
    }

    return s_mode;


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

void DeckLinkOutputPort::auto_dectect_mode(IDeckLinkVideoFrame* vframe)
{
    int tWidth, tHeight;

    tWidth = vframe->GetWidth();
    tHeight = vframe->GetHeight();

    if (resChanged(vframe))
    {

        dlbool_t pb_running;
        CHECK_DECK_ERROR(output->IsScheduledPlaybackRunning(&pb_running));

        if (pb_running)
        {
            CHECK_DECK_ERROR(output->StopScheduledPlayback(getCurrentPBTime(), NULL, 50000));
        }

        if (tWidth == 1920)
        {
            selectedMode = getModeIndex(HD_MODE);
            displayMode = displayModes[selectedMode];
            width = tWidth;
            height = tHeight;
            CHECK_DECK_ERROR(output->DisableVideoOutput());
            configure();
        }
        else {
            selectedMode = getModeIndex(HD_MODE);
            displayMode = displayModes[selectedMode];
            width = tWidth;
            height = tHeight;
            CHECK_DECK_ERROR(output->DisableVideoOutput());
            configure();
        }
    }

    IDeckLinkDisplayMode* d_mode = displayModes[selectedMode];
    result = output->CreateVideoFrame(
        d_mode->GetWidth(),
        d_mode->GetHeight(),
        ((d_mode->GetWidth() + 47) / 48) * 128,
        bmdFormat10BitYUV,
        bmdFrameFlagDefault,
        &frame);

}

bool DeckLinkOutputPort::resChanged(IDeckLinkVideoFrame* vFrame)
{
    int vWidth, vHeight;
    vWidth = vFrame->GetWidth();
    vHeight = vFrame->GetHeight();
    return !(vWidth == width && height == vHeight);
}

void DeckLinkOutputPort::synchronize(bool* _sync)
{
    _release_frames = _sync;
}

bool DeckLinkOutputPort::waitForReference()
{
    IDeckLinkStatus* m_decklinkStatus;
    // wait indefinitely for the reference signal, or be cancel and go on free run.
    result = port->QueryInterface(IID_IDeckLinkStatus, (void**)&m_decklinkStatus);
    if (result != S_OK)
    {
        return false;
    }

    dlbool_t referenceLocked;
    if (!m_decklinkStatus)
        return false;

    std::cout << "[info]: Waiting for reference signal..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    while (true)
    {
        if (m_decklinkStatus->GetFlag(bmdDeckLinkStatusReferenceSignalLocked, &referenceLocked) == S_OK && referenceLocked)
        {
            m_referenceLocked = true;
            return true;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        std::cout << "[info]: Waiting for Gen lock" << std::endl;
        /*  if ((std::chrono::high_resolution_clock::now() - start) >= std::chrono::seconds(5))
              break;*/
    }

    this->m_referenceLocked = false;
    return false;
}


IDeckLinkMutableVideoFrame* DeckLinkOutputPort::get_Fill_frame(int w, int h)
{
    IDeckLinkMutableVideoFrame* frame;

    result = output->CreateVideoFrame(w, h, w * 4, bmdFormat8BitBGRA, bmdFrameFlagDefault, &frame);

    if (result == S_OK)
        return frame;
    return nullptr;
}

IDeckLinkMutableVideoFrame* DeckLinkOutputPort::get_Key_frame(int w, int h)
{
    IDeckLinkMutableVideoFrame* frame;

    result = output->CreateVideoFrame(w, h, w * 2, bmdFormat8BitYUV, bmdFrameFlagDefault, &frame);

    if (result == S_OK)
        return frame;

    return nullptr;
}

IDeckLinkMutableVideoFrame* DeckLinkOutputPort::get_mutable_frame(int w, int h)
{
    IDeckLinkMutableVideoFrame* frame;

    result = output->CreateVideoFrame(
        w,
        h,
        ((w + 47) / 48) * 128,
        bmdFormat10BitYUV,
        bmdFrameFlagDefault,
        &frame);

    if (result == S_OK)
        return frame;
    return nullptr;
}

BMDTimeValue DeckLinkOutputPort::getCurrentPBTime()
{
    if (cb)
        return cb->getCurrentDisplayTime();
}

void DeckLinkOutputPort::run()
{
    waitForReference(); // this will wait for gen_lock, before starting the playback.
    std::cout << "Decklink Frame: " << frame->GetRowBytes() << std::endl;
    while (running)
    {
        if (!m_referenceLocked)
            std::cout << "[info]: Waiting for Gen lock" << std::endl;
        // if we are synchronized, wait for the flag .... or else ... just let it go.
        if ((_release_frames != nullptr && *_release_frames) || (!_release_frames))
        {
            if (frames_q != nullptr && !frames_q->empty())
            {
                IDeckLinkMutableVideoFrame* iframe = (IDeckLinkMutableVideoFrame*)frames_q->front();
                if (!frame)
                {
                    IDeckLinkDisplayMode* d_mode = displayModes[selectedMode];
                    result = output->CreateVideoFrame(
                        d_mode->GetWidth(),
                        d_mode->GetHeight(),
                        ((d_mode->GetWidth() + 47) / 48) * 128,
                        bmdFormat10BitYUV,
                        bmdFrameFlagDefault,
                        &frame);
                }
                // now convert frame from bgra to YUV
                if (!conversion)
                {
                    CHECK_DECK_ERROR(GetDeckLinkFrameConverter(&conversion));
                    CHECK_DECK_ERROR(conversion->ConvertFrame(iframe, frame));
                }
                else {
                    CHECK_DECK_ERROR(conversion->ConvertFrame(iframe, frame));
                }

                //playFrameBack();
                frames_q->pop(); 
            }
        }
    }
}

void CameraOutputPort::run()
{
    //waitForReference(); // this will wait for gen_lock, before starting the playback.

    IDeckLinkDisplayMode* d_mode = displayModes[selectedMode];
    result = output->CreateVideoFrame(
        d_mode->GetWidth(),
        d_mode->GetHeight(),
        (((d_mode->GetWidth() + 47) / 48) * 128),
        bmdFormat10BitYUV,
        bmdFrameFlagDefault,
        &frame);

    IDeckLinkMutableVideoFrame* srcFrame;

    result = output->CreateVideoFrame(
        d_mode->GetWidth(),
        d_mode->GetHeight(),
        (d_mode->GetWidth() * 2),
        bmdFormat8BitYUV,
        bmdFrameFlagDefault,
        &srcFrame
    );
    
    BMDTimeValue frame_duration;
    BMDTimeScale d_scale;

    d_mode->GetFrameRate(&frame_duration, &d_scale);

    int frame_rate = d_scale / frame_duration;
    int m_seconds = (1000 / frame_rate);

    while (running)
    {
        // if we are synchronized, wait for the flag .... or else ... just let it go.
        if ((_release_frames != nullptr && *_release_frames) || (!_release_frames))
        {
            if (frames_q && !frames_q->empty())
            {
                IDeckLinkMutableVideoFrame* vFrame = (IDeckLinkMutableVideoFrame*) frames_q->front();
                void* buffer = nullptr;

                if (!conversion)
                {
                    CHECK_DECK_ERROR(GetDeckLinkFrameConverter(&conversion));
                    CHECK_DECK_ERROR(conversion->ConvertFrame(vFrame, frame));
                }
                else {
                    CHECK_DECK_ERROR(conversion->ConvertFrame(vFrame, frame));
                }


                //if (vFrame)vFrame->GetBytes(&buffer);
              /*  if (buffer)
                {
                    this->add_to_q(buffer, false);
                }*/
                vFrame->Release();
            }
            vMtx.lock();
            playFrameBack();
            vMtx.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(m_seconds)); // need to implement hardware reference clock here...
        }
    }
}

void CameraOutputPort::add_to_q(void* data, bool clear)
{
    vMtx.lock();
    void* buffer;
    frame->GetBytes(&buffer);
   
    if (data)
        memcpy(buffer, data, frame->GetRowBytes() * frame->GetHeight());

    if (clear)
        free(data);
    vMtx.unlock();
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

std::queue<IDeckLinkVideoFrame*>* DeckLinkOutputPort::get_output_q()
{
    if (!frames_q)
        frames_q = new std::queue<IDeckLinkVideoFrame*>();

    return frames_q;
}

void DeckLinkOutputPort::AddFrame(void* frameBuffer, size_t size)
{
    if (!srcFrame)
    {
        IDeckLinkDisplayMode* d_mode = displayModes[selectedMode];

        result = output->CreateVideoFrame(
            d_mode->GetWidth(),
            d_mode->GetHeight(),
            pixelFormat == bmdFormat8BitBGRA ? d_mode->GetWidth() * 4 : d_mode->GetWidth() * 2,
            pixelFormat,
            bmdFrameFlagDefault,
            &srcFrame);

        if (result != S_OK)
        {
            frame = nullptr;
            return;
        }
        else {
            uchar* buffer;
            srcFrame->GetBytes((void**)&buffer);
            memcpy(buffer, frameBuffer, size);
        }

    }
    else {
        uchar* buffer;
        srcFrame->GetBytes((void**)&buffer);
        memcpy(buffer, frameBuffer, size);
    }

    if (frames_q == nullptr)
    {
        frames_q = new std::queue<IDeckLinkVideoFrame*>();
        frames_q->push(srcFrame);
    }
    else {

        frames_q->push(srcFrame);
    }

    if (!running)
        start();

    //this->output->DisplayVideoFrameSync(frame);
}

void DeckLinkOutputPort::playFrameBack()
{
    BOOL playback_running;

    this->output->IsScheduledPlaybackRunning(&playback_running);
    static bool _ran = false;
    unsigned int buffered_frames = 0;

    static BMDTimeValue tv = (buffered_frames * 1000), duration = 1000;
    static BMDTimeScale scale = 50000;

    if (!playback_running)
    {
        if (!_ran)
        {
            displayMode->GetFrameRate(&duration, &scale);
            if (output->GetBufferedVideoFrameCount(&buffered_frames) == S_OK)
            {
                tv = (buffered_frames * 1000);
            }

            cb->setDuration(duration);
            cb->setTimeScale(scale);
            cb->setTimeValue(0);


            this->output->SetScheduledFrameCompletionCallback(cb);
            _ran = true;
        }
        else {

            if (output->GetBufferedVideoFrameCount(&buffered_frames) == S_OK)
            {
                tv = (buffered_frames * duration);
            }
            this->output->ScheduleVideoFrame(frame, tv, duration, scale);
        }

        if (buffered_frames >= PREROLL)
            this->output->StartScheduledPlayback(0, scale, 1.0);
    }
    else {

        cb->addFrame(frame);
    }
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
    BOOL pb_running;
    BMDTimeValue actual_stop_time;
    if (S_OK == output->IsScheduledPlaybackRunning(&pb_running))
    {
        if (pb_running)
        {
            output->StopScheduledPlayback(cb->getCurrentDisplayTime(), &actual_stop_time, cb->getTimeScale());
        }
    }

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
    deckLinkCap->startCapture(displayModes[selectedMode]->GetDisplayMode(), nullptr, true);
}
void DeckLinkInputPort::subscribe_2_input_q(std::queue<IDeckLinkVideoInputFrame*>* q)
{
    assert(q);
    callback->subscribe_2_q(q);
}

void DeckLinkInputPort::subscribe_2_input_q(std::queue<IDeckLinkVideoFrame*>* q)
{
    assert(q);
    callback->subscribe_2_q(q);
}

DeckLinkPlaybackCallback::DeckLinkPlaybackCallback(IDeckLinkOutput* dev)
    : m_port(dev),
    count(0),
    scale(50000),
    f_duration(1000),
    frames_count(0),
    timeValue(0)
{}

HRESULT DeckLinkPlaybackCallback::ScheduledFrameCompleted(IDeckLinkVideoFrame* completedFrame, BMDOutputFrameCompletionResult result)
{
    return S_OK;
}

void DeckLinkPlaybackCallback::addFrame(IDeckLinkVideoFrame* frame)
{
    if (timeValue == 0)
    {
        timeValue += PREROLL * f_duration - f_duration;
    }

    timeValue += f_duration;
    HRESULT result = m_port->ScheduleVideoFrame(frame, timeValue, f_duration, scale);

    if (S_OK != result)
    {
        switch (result)
        {
        case E_FAIL:
            std::cout << "There was a failure: " << frame << std::endl;
            break;
        case E_ACCESSDENIED:
            std::cout << "Access denied (Please enable video out)" << std::endl;
            break;

        case E_INVALIDARG:
            std::cout << "The frame attributes are invalid" << std::endl;
            break;

        case E_OUTOFMEMORY:
            std::cout << "Out of memory" << std::endl;
            break;
        }
    }

    /* unsigned int b_count;
      m_port->GetBufferedVideoFrameCount(&b_count);
      std::cout << "Frames Q:" << b_count << std::endl;*/
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
    frames_queue(nullptr),
    frames_q(nullptr)
{}
// This is called on a seperate thread ...
void VideoFrameCallback::arrived(IDeckLinkVideoInputFrame* frame)
{
    start_clock = std::chrono::high_resolution_clock::now();

    // std::cout << "Frame Arrival Difference: " << (start_clock - stop_clock).count() / 1000000.0 << " ms" << std::endl;
   
    frame->AddRef();
    // interogate the frame to decide how to process it ...
    width = frame->GetWidth();
    height = frame->GetHeight();

    switch (frame->GetPixelFormat())
    {
    case bmdFormat10BitYUV:
    {
        
        if (frame->GetRowBytes() * frame->GetHeight() < (1920 * 1080))
            return;

        //std::cout << "Received Res: " << frame->GetWidth() << " x " << frame->GetHeight() << std::endl;
        if (frames_q)
            frames_q->push(frame); // generic frames q for Decklink Video Frames

        if (frames_queue)
            frames_queue->push(frame);
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
    stop_clock = std::chrono::high_resolution_clock::now();
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

void VideoFrameCallback::subscribe_2_q(std::queue<IDeckLinkVideoFrame*>* q)
{
    frames_q = q;
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