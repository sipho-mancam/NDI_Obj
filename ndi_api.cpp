// NDI_Obj.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "ndi_api.hpp"

void init()
{
    if (NDIlib_is_supported_CPU()) {
        assert(NDIlib_initialize());
    }
    else {
        std::cerr << "CPU doesn't support NDI Library" << std::endl;
        throw(NDI_exception("Couldn't initialize NDI .... exiting."));
    }
}

void clean_up()
{
    NDIlib_destroy();
}

uint32_t idGene()
{
    ids++;
    return ids;
}

void resetID()
{
    ids = 0;
}


Discovery::Discovery(bool* controller, NDIlib_find_create_t desc, bool autoID)
    : NDI_Obj(controller), status(true), current_sources_count(0), sources(nullptr), threadI(nullptr), save(true), time_out(1000), running(false)
{
    selected_device = "";
    disc_instance = NDIlib_find_create_v2(&discovery_description);
    if (!disc_instance)
    {
        throw(NDI_exception("Failed to create NDI disovery object"));
    }
}

void Discovery::run() // run the discovery until we are told to stop
{

    while (!(*exit) && status)
    {
        NDIlib_find_wait_for_sources(disc_instance, time_out);
        sources = (NDIlib_source_t*)NDIlib_find_get_current_sources(disc_instance, &current_sources_count);

        if (current_sources_count)
            discovered_sources.clear();

        for (unsigned int i = 0; i < current_sources_count; i++)
        {
            discovered_sources.insert(std::string(sources[i].p_ndi_name)); // the discovered sources to the set
            if (save)
                saved_sources.insert(std::string(sources[i].p_ndi_name));
        }
    }
}

void Discovery::start()
{
    this->status = true;
    running = true;
    threadI = new std::thread(&Discovery::run, this);
}

void Discovery::stop() {
    this->status = false;
    running = false;
    if (threadI != nullptr)
    {
        threadI->join();
        delete threadI;
        threadI = nullptr;
    }
}

void Discovery::showMeList()
{
    frozen_list.clear(); // clean the frozen least, so that it's consistent with the display ...

    system("cls");
    printf("\n\tNDI sources on the Network %s\n", selected_device.empty() ? "" : std::string("--> " + selected_device).c_str());
    printf("\t-----------------------------------------\n");
    int i = 0;
    for (std::string s : this->discovered_sources)
    {
        frozen_list.push_back(s);
        printf("%d.\t%s\n", i, s.c_str());
        i++;
    }
}

std::string Discovery::selectDevice(int s)
{
    if (s < 0 || s >= frozen_list.size()) return "";

    selected_device = frozen_list[s];
    return selected_device;
}

std::set<std::string> Discovery::getDiscoveredSources() // this will wait for sources ... blocking call
{
    while (running && current_sources_count == 0)
        std::this_thread::sleep_for(std::chrono::milliseconds(time_out));
    return discovered_sources;
}

Discovery::~Discovery()
{
    this->stop();
    NDIlib_find_destroy(this->disc_instance);
}


void NDI_Recv::run()
{

    if (!frames)
        frames = new std::queue<NDIlib_video_frame_v2_t*>();

    cv::Mat preview;
    while (!(*exit) && running)
    {
        // The descriptors
        NDIlib_video_frame_v2_t video_frame;
        NDIlib_audio_frame_v2_t audio_frame;
        NDIlib_metadata_frame_t metadata_frame;

        switch (NDIlib_recv_capture_v2(rec_instance, &video_frame, NULL, &metadata_frame, timeout)) {
            // No data
        case NDIlib_frame_type_none:
            break;
            // Video data
        case NDIlib_frame_type_video:
        {
            if (fillAndKey && fillPort != nullptr && keyPort != nullptr)
            {
                uint* yuvFill;
                uchar* gBgra;
                gBgra = get_yuv_from_bgr_packed(video_frame.xres, video_frame.yres, video_frame.p_data, &yuvFill);

                uchar* alpha_channel;
                get_alpha_channel_gpu(video_frame.xres, video_frame.yres, gBgra, &alpha_channel);

                uint* key_packed;
                alpha_2_decklink_gpu(video_frame.xres, video_frame.yres, alpha_channel, &key_packed);

                fillPort->AddFrame(yuvFill, sizeof(uint) * (video_frame.xres / 2) * (video_frame.yres));

                keyPort->AddFrame(key_packed, sizeof(uint) * (video_frame.xres / 2) * (video_frame.yres));

                cudaFreeHost(key_packed);
                cudaFreeHost(yuvFill);

                NDIlib_recv_free_video_v2(rec_instance, &video_frame);
                continue;
            }
            if (persFrame->line_stride_in_bytes != video_frame.line_stride_in_bytes)
            {
                persFrame->xres = video_frame.xres;
                persFrame->yres = video_frame.yres;
                persFrame->data_size_in_bytes = video_frame.data_size_in_bytes;
                persFrame->line_stride_in_bytes = video_frame.line_stride_in_bytes;
                persFrame->FourCC = video_frame.FourCC;
                persFrame->frame_format_type = video_frame.frame_format_type;
                persFrame->timecode = video_frame.timecode;
                persFrame->timestamp = video_frame.timestamp;
                persFrame->p_metadata = video_frame.p_metadata;
                persFrame->frame_rate_D = video_frame.frame_rate_D;
                persFrame->frame_rate_N = video_frame.frame_rate_N;
                persFrame->picture_aspect_ratio = video_frame.picture_aspect_ratio;

                if (persFrame->p_data)
                    delete persFrame->p_data; 
                persFrame->p_data = nullptr;
            }
            

            if(persFrame->p_data == nullptr)
                persFrame->p_data = (uint8_t*) new uint8_t[video_frame.line_stride_in_bytes * video_frame.yres];

            memcpy(persFrame->p_data, video_frame.p_data, video_frame.line_stride_in_bytes * video_frame.yres);

            if(frames)
                frames->push(persFrame);
            // Double check that we are running sufficiently well
            //NDIlib_recv_queue_t recv_queue;
            //NDIlib_recv_get_queue(rec_instance, &recv_queue);
            //if (recv_queue.video_frames > 2) {
            //    // Display the frames per second
            //    printf("Channel %d queue depth is %d.\n", channel, recv_queue.video_frames);
            //}

            NDIlib_recv_free_video_v2(rec_instance, &video_frame);

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            
            break;
        }

        // Meta data
        case NDIlib_frame_type_metadata:
            //printf("Length: %d\nMeta: %s \n", metadata_frame.length, metadata_frame.p_data);
            NDIlib_recv_free_metadata(rec_instance, &metadata_frame);
            break;
            // There is a status change on the receiver (e.g. new web interface)
        case NDIlib_frame_type_status_change:
            printf("Receiver connection status changed.\n");
            break;
            // Everything else
        default:
            break;
        }
    }
}


void NDI_Recv::splitKeyandFill(cv::Mat& src, cv::Mat& dstA, cv::Mat& dstB /*This must be the alpha channel*/)
{
    // assuming dstA and dstB are pre-allocated ... split the channels...
    cv::Mat out[] = { dstB };
    int from_to[] = { 3,0 };
    cv::mixChannels(&src, 1, out, 1, from_to, 1);
}


NDI_Recv::NDI_Recv(bool* controller, uint32_t c, std::string s)
    : channel(c), source(s),
    NDI_Obj(controller),
    receiver_thread(nullptr),
    frames(nullptr),
    delay(3),
    connected(false),
    fillAndKey(false),
    running(false),
    fillPort(nullptr),
    keyPort(nullptr), 
    persFrame(nullptr)
{

    persFrame = new NDIlib_video_frame_v2_t(0,0);
    persFrame->p_data = nullptr;
    persFrame->xres = 0;
    persFrame->yres = 0;
    persFrame->data_size_in_bytes = 0;
    persFrame->line_stride_in_bytes = 0;
    persFrame->timecode = 0;
    persFrame->timestamp = 0;
    persFrame->p_metadata = 0;
    persFrame->frame_rate_D = 0;
    persFrame->frame_rate_N = 0;
    persFrame->picture_aspect_ratio = 0;


    if (channel == -1)
        channel = id + 10;

    if (!s.empty())
    {
        s_connect.p_ndi_name = source.c_str();

        recv_desc.bandwidth = NDIlib_recv_bandwidth_highest;
        recv_desc.source_to_connect_to = s_connect; // this will allow the NDI endpoint connect to the specified string upon creation...
        recv_desc.color_format = NDIlib_recv_color_format_best;

        char buf[256] = { 0, };
        sprintf_s(buf, "Channel %u", channel);
        recv_desc.p_ndi_recv_name = buf;



        if (!rec_instance)
        {
            throw(NDI_exception("Failed to create receiver."));
        }
        connected = true;
    }
    rec_instance = NDIlib_recv_create_v3(&recv_desc);
}

void NDI_Recv::disconnect()
{
    connected = false;
    NDIlib_recv_connect(rec_instance, NULL);
}

void NDI_Recv::setKeyAndFillPorts(DeckLinkOutputPort* f, DeckLinkOutputPort* k)
{
    this->keyPort = k;
    this->fillPort = f;
    //this->fillPort->SetPixelFormat(bmdFormat8BitBGRA);
}

void NDI_Recv::connect(std::string s)
{
    if (s.empty())return;
    // disconnect first
    NDIlib_recv_connect(rec_instance, NULL);
    source = s;
    s_connect.p_ndi_name = source.c_str();
    // connect to the new source
    NDIlib_recv_connect(rec_instance, &s_connect);

    if (!rec_instance)
        throw(NDI_exception("Failed to create receiver."));
    connected = true;
    std::cout << "Connected ..." << std::endl;
}

void NDI_Recv::subscribe_to_q(std::queue< NDIlib_video_frame_v2_t*>* qu)
{
    frames = qu;
}

void NDI_Recv::start()
{
    running = true;
    if (connected)
        this->receiver_thread = new std::thread(&NDI_Recv::run, this);
}

void NDI_Recv::stop()
{
    if (running)
    {
        running = false;
        this->receiver_thread->join();
        delete this->receiver_thread;
    }
}

void NDI_Recv::popFrame()
{
    NDIlib_recv_free_video_v2(rec_instance, (NDIlib_video_frame_v2_t*)&frames->front());
    frames->pop();
}

void NDI_Recv::clearAll()
{
    while (frames->size() > 0)
    {
        NDIlib_recv_free_video_v2(rec_instance, (NDIlib_video_frame_v2_t*)&frames->front());
        frames->pop();
    }
}

NDI_Recv::~NDI_Recv()
{
    this->stop();
    NDIlib_recv_destroy(rec_instance);
}



NDI_Sender::NDI_Sender(bool* controller, std::string source)
    : NDI_Obj(controller), sender(NULL), p_worker(nullptr), init_d(false), running(false), frames_q(nullptr)
{
    desc.clock_video = true;
    desc.p_ndi_name = "Decklink_Viz_bridge";
    desc.p_groups = NULL;

    sender = NDIlib_send_create(&desc);

    assert(sender != NULL);
}

void NDI_Sender::start()
{
    running = true;
    if (p_worker)
        delete p_worker;
    p_worker = new std::thread(&NDI_Sender::run, this);

    if (!p_worker)
        running = false;
}

void NDI_Sender::stop()
{
    running = false;
    if (p_worker)
    {
        p_worker->join();
        delete p_worker;
        p_worker = nullptr;
    }
}

void NDI_Sender::run()
{
    while (!(*this->exit) && running)
    {
        if (frames_q && !frames_q->empty())
        {
            NDI_video_frame_16bit = frames_q->front();
            NDIlib_send_send_video_v2(sender, &NDI_video_frame_16bit);
            frames_q->pop();
            free(NDI_video_frame_16bit.p_data);

        }
        
    }
}

void NDI_Sender::subscribe_to_q(std::queue<NDIlib_video_frame_v2_t>* q)
{
    if (q)
    {
        if (frames_q)
        {
            while (!frames_q->empty())
                frames_q->pop();
        }
        frames_q = q;
    }
}

NDI_Sender::~NDI_Sender()
{
    this->stop();
    // free some other buffers here
    NDIlib_send_destroy(sender);
}