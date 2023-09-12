#include "ndi_api.hpp"
#include "interface_manager.hpp"
#include "decklink_api.hpp"
#include "internal_sync.hpp"

#include <fstream>
#include <iostream>
#include <ctime>

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
    bool exit_flag = false;
    int disp_mode = 0; // change the mode to either HD or UHD mode (1 = UHD, 0 = HD)
    
    int console_key = 0, choice = 0;
    /*std::cout << "\n   ======= Select Mode =======\n\n0. HD\t(1920 x 1080 --- 1080i50)\n1. UHD\t(3840 x 2160 --- 2160p50)\nChoice: ";
    std::cin >> disp_mode;
    if (disp_mode != 1 && disp_mode != 0)
        disp_mode = 1;*/

    system("cls");
    DeckLinkCard* card = new DeckLinkCard();
    DeckLinkOutputPort* fillPort = card->SelectOutputPort(3, disp_mode);
    DeckLinkOutputPort* keyPort = card->SelectOutputPort(1, disp_mode);

    if (fillPort == nullptr || keyPort == nullptr)
    {
        std::cout << "Key and Fill Ports could not be selected" << std::endl;
    }
    /*
    std::cout << "[info]: Using DeckLink (4) and DeckLink (3) as Key and Fill Ports." << std::endl;
    std::cout << "Press any key to continue ..." << std::endl;
    _getch();*/

    DeckLinkInputPort* camera_input = card->SelectInputPort(0);
    CameraOutputPort* camera_output = card->SelectCamOutputPort(2, disp_mode);
    camera_input->subscribe_2_input_q(camera_output->get_output_q());

    Synchronizer frames_synchronizer;  
    frames_synchronizer.add_output(fillPort);
    frames_synchronizer.add_output(keyPort);
    frames_synchronizer.start();
    
    NDI_Key_And_Fill* key_and_fill = new NDI_Key_And_Fill(&exit_flag, 1, "");
    key_and_fill->setKeyAndFillPorts(fillPort, keyPort);
     
    Discovery* discovery = new Discovery(&exit_flag);
    discovery->start();
    discovery->showMeList();


    auto start = std::chrono::high_resolution_clock::now();
    while (!exit_flag)
    {

        if (((std::chrono::high_resolution_clock::now() - start) >= std::chrono::seconds(1)))
        {
            // update the user list
            discovery->showMeList(disp_mode);
            start = std::chrono::high_resolution_clock::now();
        }

        if (_kbhit()) {
            console_key = _getch();
            if (console_key == 27)
                exit_flag = true;

            switch (console_key)
            {
            case 's':
            {

                key_and_fill->stop();
                system("cls");

                // show the most recent list...
                discovery->showMeList(disp_mode);
                std::cout << "Selected Device using index (0, 1, 2 ...etc): ";
                std::cin >> choice;
                std::string s = discovery->selectDevice(choice);

                if (s.empty())
                {
                    std::cout << "Index out of range ..." << std::endl;
                    _getch();
                }
                else {
                    key_and_fill->connect(s);
                }

                key_and_fill->start();
              /*  camera_input->startCapture();
                camera_output->start();*/
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
    
    delete key_and_fill;
    delete discovery;
    delete card;
    clean_up();
}




