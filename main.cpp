#include "ndi_api.hpp"
#include "interface_manager.hpp"
#include "decklink_api.hpp"
#include "internal_sync.hpp"

#include <fstream>
#include <iostream>
#include <ctime>
#include <Windows.h>

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

    std::cout << "Select Display Mode" << std::endl;
    std::cout << "======================" << std::endl;
    std::cout << "0. HD (1080i)\n1.UHD (2160p)" << std::endl;
    std::cout << "Option: ";
    std::cin >> disp_mode;

    if (disp_mode != 1 && disp_mode != 0)
        disp_mode = 0;


    system("cls");
    DeckLinkCard* card = new DeckLinkCard();
    DeckLinkOutputPort* fillPort = card->SelectOutputPort(3, disp_mode);
    DeckLinkOutputPort* keyPort = card->SelectOutputPort(1, disp_mode);

    if (fillPort == nullptr || keyPort == nullptr)
    {
        std::cout << "Key and Fill Ports could not be selected" << std::endl;
    }

    TCHAR compName[MAX_COMPUTERNAME_LENGTH + 3];
    DWORD size = MAX_COMPUTERNAME_LENGTH + 3;

    GetComputerName(compName, &size);
    std::string connection_string(compName);
    connection_string += " (VizEngine-0)";

    Synchronizer frames_synchronizer;  
    frames_synchronizer.add_output(fillPort);
    frames_synchronizer.add_output(keyPort);
    frames_synchronizer.start();
    
    NDI_Key_And_Fill* key_and_fill = new NDI_Key_And_Fill(&exit_flag, 1, "");
    key_and_fill->setKeyAndFillPorts(fillPort, keyPort);
    key_and_fill->connect(connection_string);
    key_and_fill->start();

    while (!exit_flag)
    {

        if (_kbhit()) {
            console_key = _getch();
            if (console_key == 27)
                exit_flag = true;
        }
    }
    
    delete key_and_fill;
    delete card;
    clean_up();
}




