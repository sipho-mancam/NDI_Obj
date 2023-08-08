// NDI_Obj.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

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

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef _WIN64
#pragma comment(lib, "Processing.NDI.Lib.x64.lib")
#else
#pragma comment(lib, "Processing.NDI.Lib.x86.lib")
#endif

//#define _DEBUG

#include <Processing.NDI.Lib.h>
#include <exception>


static uint32_t ids = 0;

uint32_t idGene()
{
    ids++;
    return ids;
}

void resetID()
{
    ids = 0;
}


class NDI_exception : public std::exception {
private:
    const char* const msg;

public:
    NDI_exception(const char* const m)
        :msg(m), exception((const char * const)msg){}

    const char* what() const override
    {
        return msg;
    }
};



class NDI_Obj {
protected:
    uint32_t id;
    bool *exit;
    uint32_t timeout;
public:
    NDI_Obj(bool* ex = nullptr) : exit(ex), id(-1), timeout(100) { id = idGene(); }
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
    Discovery(bool* controller, NDIlib_find_create_t desc = (NDIlib_find_create_t)NULL, bool autoID = true) 
        : NDI_Obj(controller), status(true), current_sources_count(0), sources(nullptr), threadI(nullptr), save(true), time_out(1000), running(false)
    {
        selected_device = "";
        disc_instance = NDIlib_find_create_v2(&discovery_description);
        if (!disc_instance)
        {
            throw(NDI_exception("Failed to create NDI disovery object"));
        }
    }

    bool getStatus() { return  status; }
    void disableSave() { this->save = false; saved_sources.clear(); }
    void enableSave() { this->save = true; }
    bool isSourcesReady() { return current_sources_count > 0; }

    void run() override // run the discovery until we are told to stop
    {
        
        while (!(*exit) && status)
        {
            NDIlib_find_wait_for_sources(disc_instance, time_out);
            sources = (NDIlib_source_t *) NDIlib_find_get_current_sources(disc_instance, &current_sources_count);

            if(current_sources_count)
                discovered_sources.clear();
            
            for (unsigned int i = 0; i < current_sources_count; i++)
            {
               discovered_sources.insert(std::string(sources[i].p_ndi_name)); // the discovered sources to the set
               if (save)
                   saved_sources.insert(std::string(sources[i].p_ndi_name));
            }   
        }
    }

    void start()
    {
        this->status = true;
        running = true;
        threadI = new std::thread(&Discovery::run, this);
#ifdef _DEBUG
        std::cout << "Started Discovery service ..." << std::endl;
#endif;
    }

    void stop() {
        this->status = false;
        running = false;
        if (threadI != nullptr)
        {
            threadI->join();
            delete threadI;
            threadI = nullptr;
        }
    }

    void showMeList()
    {
        frozen_list.clear(); // clean the frozen least, so that it's consistent with the display ...

        system("cls");
        printf("\n\tNDI sources on the Network (%s)\n", selected_device.c_str());
        printf("\t----------------------------------\n");
        int i = 0;
        for (std::string s : this->discovered_sources)
        {
            frozen_list.push_back(s);
            printf("%d.\t%s\n", i, s.c_str());
            i++;
        }
    }
    
    std::string selectDevice(int s)
    {
        if (s < 0 || s >= frozen_list.size()) return "";

        selected_device = frozen_list[s];
        return selected_device;
    }

    std::string getSelectedDevice() { return selected_device; } 




    std::set<std::string> getDiscoveredSources() // this will wait for sources ... blocking call
    {
        while (running && current_sources_count == 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(time_out));
        return discovered_sources;
    }

    std::set<std::string> getSavedSources() { return this->saved_sources; }

    ~Discovery()
    {
        this->stop();
        NDIlib_find_destroy(this->disc_instance);
    } 

    // TODO: Write the discovered sources to a mysql database for UI to update ...
};



class NDI_Recv : public NDI_Obj {
private:
    std::string source;
    uint32_t channel;
    NDIlib_source_t s_connect;
    NDIlib_recv_instance_t rec_instance;
    NDIlib_recv_create_v3_t recv_desc;
    std::thread* receiver_thread;
    uint32_t delay;
    std::queue<NDIlib_video_frame_v2_t> *frames;

public:
    NDI_Recv(bool *controller, uint32_t c, std::string s) 
        : channel(c), source(s) , NDI_Obj(controller), receiver_thread(nullptr), frames(nullptr), delay(3)
    {
        if (s.empty())
        {
            char buf[2048];
            sprintf_s(buf, "Value Error: Cannot pass empty source\nat: %s : line: %d", __FILE__, __LINE__);
            throw(NDI_exception( buf ));
        }
        s_connect.p_ndi_name = source.c_str();

        recv_desc.bandwidth = NDIlib_recv_bandwidth_highest;
        recv_desc.source_to_connect_to = s_connect;
        char buf[256] = { 0, };
        sprintf_s(buf, "Channel %u", channel);
        recv_desc.p_ndi_recv_name = buf;

        rec_instance = NDIlib_recv_create_v3(&recv_desc);

        if (!rec_instance)
        {
            throw(NDI_exception("Failed to create receiver."));
        }
    }

    void subscribeQueue(std::queue< NDIlib_video_frame_v2_t>* qu)
    {
        frames = qu;
    }

    void run() override 
    {
        if (!frames)
            frames = new std::queue<NDIlib_video_frame_v2_t>();

        while (!(*exit))
        {
            // The descriptors
            NDIlib_video_frame_v2_t video_frame;
            NDIlib_audio_frame_v2_t audio_frame;
            NDIlib_metadata_frame_t metadata_frame;

            switch (NDIlib_recv_capture_v2(rec_instance, &video_frame, NULL, &metadata_frame, timeout)) {
                // No data
            case NDIlib_frame_type_none:
                //std::cout << "No data" << std::endl;
                break;

                // Video data
            case NDIlib_frame_type_video:
            {
                printf("Channel %d %s : Video received: %u x %u\n",channel, source.c_str(), video_frame.xres, video_frame.yres);

                frames->push(video_frame);

                // Double check that we are running sufficiently well
                NDIlib_recv_queue_t recv_queue;
                NDIlib_recv_get_queue(rec_instance, &recv_queue);
                if (recv_queue.video_frames > 2) {
                    // Display the frames per second
                    printf("Channel %d queue depth is %d.\n", channel, recv_queue.video_frames);
                }

                if (frames->size() > delay)
                {
                    NDIlib_recv_free_video_v2(rec_instance, (NDIlib_video_frame_v2_t*)&frames->front());
                    frames->pop();
                }

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

    void start()
    {
        this->receiver_thread = new std::thread(&NDI_Recv::run, this);

    }

    void popFrame()
    {
        
    }

    void clearAll()
    {
        while (frames->size() > 0)
        {
            NDIlib_recv_free_video_v2(rec_instance, (NDIlib_video_frame_v2_t*)&frames->front());
            frames->pop();
        }
    }

    uint32_t getChannel() { return channel; }
    std::string getSource() { return source; }

    ~NDI_Recv()
    {
        receiver_thread->join();
        delete receiver_thread;
        NDIlib_recv_destroy(rec_instance);
    }
};


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


int main()
{
    init();

    bool exit_flag = false;
    Discovery* discovery = new Discovery(&exit_flag);

    auto start = std::chrono::high_resolution_clock::now();

    discovery->start();

    int console_key = 0, choice = 0;

    discovery->showMeList();

    while (!exit_flag)
    {
      
        if (((std::chrono::high_resolution_clock::now() - start) >= std::chrono::seconds(1)))
        {
            // update the user list
            discovery->showMeList();

            start = std::chrono::high_resolution_clock::now();
        }



        if (_kbhit()) {
            console_key = _getch();
            if (console_key == 27)
                exit_flag = true;

            switch (console_key)
            {
            case 's':
                std::cout << "Selected Device using index (0, 1, 2 ...etc): ";
                std::cin >> choice;
                std::string s = discovery->selectDevice(choice);

                if (!s.empty())
                {
                    std::cout << "\nSelected: " << s << std::endl;
                }
                else {
                    std::cout << "Index out of range ..."<<std::endl;
                }
                _getch();
                break;
            }
        }
    }

    delete discovery;

    

    clean_up();

}

