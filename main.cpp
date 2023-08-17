#include "ndi_api.hpp"
#include "interface_manager.hpp"
#include "decklink_api.hpp"



int main()
{
    init();
    bool exit_flag = false;

    Interface_Manager interface_manager;
    interface_manager.start_decklink();
    interface_manager.start_ndi();
    DeckLinkCard* card = new DeckLinkCard();

    // this configures port 0 and 1 as output and gives you a handle to it
    //DeckLinkOutputPort* fillPort = card->SelectOutputPort(0);
    //DeckLinkOutputPort* keyPort = card->SelectOutputPort(1);

    DeckLinkInputPort* inputPort = card->SelectInputPort(3);
    assert(inputPort != nullptr);
    inputPort->subscribe_2_input_q(interface_manager.getDeckLinkInputQ());
    inputPort->startCapture();

    DeckLinkOutputPort* video_out = card->SelectOutputPort(2);
    video_out->subscribe_2_q(interface_manager.getDeckLinkOutputQ());
    video_out->start();    
    
    NDI_Sender* sender = new NDI_Sender(&exit_flag, "");
    sender->subscribe_to_q(interface_manager.getNDIOutputQ());
    sender->start();

    NDI_Recv* receiver = new NDI_Recv(&exit_flag, 0);
    receiver->subscribe_to_q(interface_manager.getNDIInputQ());
    //receiver->setKeyAndFillPorts(fillPort, keyPort);
    //receiver->enableFillAndKey();

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
    //delete video_out;
    delete receiver;
    delete sender;
    delete card;


    clean_up();
}




