#include "ndi_api.hpp"
#include "interface_manager.hpp"
#include "decklink_api.hpp"


#include <fstream>
#include <iostream>
#include <ctime>

static IDeckLinkOutput* outDevice;
HRESULT InputLoopThrough(NDI_Recv* input_source);

void log_to_file()
{
    std::ofstream log_file("C:\\Users\\Chroma2\\Documents\\ndi_deck_log.txt");

    auto start = std::chrono::high_resolution_clock::now();
    char buf[256];
    while (true)
    {
      
        if ((std::chrono::high_resolution_clock::now() - start) >= std::chrono::seconds(1))
        {
            time_t curr_time;
            curr_time = time(NULL);

            tm* tm_local = localtime(&curr_time);

            sprintf(buf, "[ %d: %d: %d]: Time log.", tm_local->tm_hour, tm_local->tm_min, tm_local->tm_sec);
            
            log_file << buf << std::endl;
            memset(buf, 0, 256);

            start = std::chrono::high_resolution_clock::now();
        } 
    }
}



int main()
{
    init();
    bool exit_flag = false, outputStarted = false;

    Interface_Manager interface_manager;
    interface_manager.start_decklink();
    //interface_manager.start_ndi();

    DeckLinkCard* card = new DeckLinkCard();

    DeckLinkInputPort* inputPort = card->SelectInputPort(0);
    assert(inputPort != nullptr);
    inputPort->subscribe_2_input_q(interface_manager.getDeckLinkInputQ());
    inputPort->startCapture();
    
    CameraOutputPort* video_out = card->SelectCamOutputPort(2, 0);
    outDevice = video_out->getOutputDevice();
    
    NDI_Sender* sender = new NDI_Sender(&exit_flag, "");
    sender->subscribe_to_q(interface_manager.getNDIOutputQ());
    sender->start();

    NDI_Recv* receiver = new NDI_Recv(&exit_flag, 0);
    receiver->subscribe_to_q(interface_manager.getNDIInputQ());
    auto start = std::chrono::high_resolution_clock::now();

    TCHAR compName[MAX_COMPUTERNAME_LENGTH + 3];
    DWORD size = MAX_COMPUTERNAME_LENGTH + 3;

    GetComputerName(compName, &size);
    std::string connection_string(compName);
    connection_string += " (VizEngine-0)";

    receiver->connect(connection_string);
    receiver->start();

    InputLoopThrough(receiver);
    //std::thread outputRender(&InputLoopThrough, receiver);

    Discovery* discovery = new Discovery(&exit_flag);
    discovery->start();
    discovery->showMeList();
    discovery->stop();

    int console_key = 0, choice = 0;

    while (!exit_flag)
    {
        if (((std::chrono::high_resolution_clock::now() - start) >= std::chrono::seconds(1)))
        {
            // update the user list
            discovery->showMeList();
            start = std::chrono::high_resolution_clock::now();
           
        }

        if (_kbhit()) 
        {
            console_key = _getch();
            if (console_key == 27)
                exit_flag = true;

            switch (console_key)
            {
            case 's':
            {

                receiver->stop();
                system("cls");
                // show the most recent list...
                discovery->showMeList();
                std::cout << "Selected Device using index (0, 1, 2 ...etc): ";
                std::cin >> choice;
                std::string s = discovery->selectDevice(choice);

                if (s.empty())
                {
                    std::cout << "Index out of range ..." << std::endl;
                    _getch();
                }
                else {
                    receiver->connect(s);
                }
                receiver->start();
                std::cin.clear();
                discovery->stop();
                break;
            }

            case 'v': // view the selected device
            {
                system("Cls");
                printf("Selected Device\n-----------------\n");
                std::cout << discovery->getSelectedDevice() << std::endl;
                std::cout << "\n\n Press any key ..." << std::endl;
                _getch();
                break;
            }
            }
        }
    }
    delete discovery;
    delete video_out;
    delete receiver;
    delete sender;
    delete card;
    //logger.join()

    clean_up();
}



IDeckLinkVideoFrame* Interface_Manager::convert_ndi_2_decklink_frame_s(NDIlib_video_frame_v2_t* ndi_frame)
{
    // Declare some decklink object and allocate it some memory .... copy video parameters and return it
    IDeckLinkMutableVideoFrame* frame = nullptr, * outframe = nullptr;
    IDeckLinkVideoConversion* converter = nullptr;
    static IDeckLinkMutableVideoFrame* frame_8 = nullptr, *frame_10=nullptr;

    extern IDeckLinkOutput* outDevice;

    if (outDevice) {
        if (frame_10 == nullptr)
        {
            CHECK_DECK_ERROR(outDevice->CreateVideoFrame(
                ndi_frame->xres,
                ndi_frame->yres,
                (((ndi_frame->xres + 47) / 48) * 128),
                bmdFormat10BitYUV,
                bmdFrameFlagDefault, &frame_10));
        }
        
    }

    void* buffer = nullptr;


    CHECK_DECK_ERROR(GetDeckLinkFrameConverter(&converter));

    switch (ndi_frame->FourCC)
    {
    case NDIlib_FourCC_type_BGRA: // 8bit BGRA data out..
    {

        if (outDevice)
        {
            if(frame_8 == nullptr)
            CHECK_DECK_ERROR(outDevice->CreateVideoFrame(
                ndi_frame->xres,
                ndi_frame->yres,
                ndi_frame->line_stride_in_bytes,
                bmdFormat8BitBGRA,
                bmdFrameFlagDefault, &frame_8));
        }
        if (frame_8) {
            frame_8->GetBytes(&buffer);
            memcpy(buffer, ndi_frame->p_data, ndi_frame->yres * ndi_frame->line_stride_in_bytes);
        }

        if (converter)
        {
            converter->ConvertFrame(frame_8, frame_10);
        }
        outframe = frame_10;
        break;
    }

    case NDIlib_FourCC_type_UYVY:
    {
        if (outDevice)
        {
            if(frame_8 ==nullptr)
            CHECK_DECK_ERROR(outDevice->CreateVideoFrame(
                ndi_frame->xres,
                ndi_frame->yres,
                ndi_frame->line_stride_in_bytes,
                bmdFormat8BitYUV,
                bmdFrameFlagDefault, &frame_8));
        }

        if (frame_8) {
            frame_8->GetBytes(&buffer);
            memcpy(buffer, ndi_frame->p_data, ndi_frame->yres * ndi_frame->line_stride_in_bytes);
        }

        if (converter)
        {
            converter->ConvertFrame(frame_8, frame_10);
        }
        outframe = frame_10;
        break;
    }
    }
    //outframe->AddRef();
    return outframe;
}




