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

    DeckLinkCard* card = new DeckLinkCard();
    DeckLinkOutputPort* fillPort = card->SelectOutputPort(3, disp_mode);
    DeckLinkOutputPort* keyPort = card->SelectOutputPort(1, disp_mode);
    DeckLinkInputPort* camera_input = card->SelectInputPort(0);
    CameraOutputPort* camera_output = card->SelectCamOutputPort(2, disp_mode);
    camera_input->subscribe_2_input_q(camera_output->get_output_q());

    Synchronizer frames_synchronizer;
    frames_synchronizer.add_output(fillPort);
    frames_synchronizer.add_output(keyPort);
    frames_synchronizer.add_output(camera_output);

    frames_synchronizer.start();
    
    NDI_Key_And_Fill* key_and_fill = new NDI_Key_And_Fill(&exit_flag, 1, "");
    key_and_fill->setKeyAndFillPorts(fillPort, keyPort);
     
    auto start = std::chrono::high_resolution_clock::now();

    Discovery* discovery = new Discovery(&exit_flag);
    discovery->start();
    discovery->showMeList();

    int console_key = 0, choice = 0;

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
            {

                key_and_fill->stop();
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
                    key_and_fill->connect(s);
                }

                key_and_fill->start();
                camera_input->startCapture();
                camera_output->start();
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
    
    delete key_and_fill;
    delete discovery;
    delete card;
   
    clean_up();
}




