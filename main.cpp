#include "ndi_api.hpp"
#include "interface_manager.hpp"
#include "decklink_api.hpp"



int main()
{
    init();
    bool exit_flag = false;
    DeckLinkCard* card = new DeckLinkCard();

    // this configures port 0 and 1 as output and gives you a handle to it
    DeckLinkOutputPort* fillPort = card->SelectOutputPort(0);
    DeckLinkOutputPort* keyPort = card->SelectOutputPort(1);

    // this configures port 2 as input and gives you a handle to it.
    // ports are zero indexed.
    DeckLinkInputPort* inputPort = card->SelectInputPort(3);

    assert(fillPort != nullptr);
    assert(keyPort != nullptr);
    assert(inputPort != nullptr);

    inputPort->startCapture();
    NDI_Sender* sender = new NDI_Sender(&exit_flag, "");

    sender->subscribe_to_q(inputPort->getQRef());

    sender->start();

    Discovery* discovery = new Discovery(&exit_flag);

    NDI_Recv* receiver = new NDI_Recv(&exit_flag, 0);

    receiver->setKeyAndFillPorts(fillPort, keyPort);

    receiver->enableFillAndKey();

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
            {

                receiver->stop();
                system("cls");
                discovery->showMeList();
                std::cout << "Selected Device using index (0, 1, 2 ...etc): ";
                std::cin >> choice;
                std::string s = discovery->selectDevice(choice);

                if (s.empty())
                {
                    std::cout << "Index out of range ..." << std::endl;
                    _getch();
                }

                if (!s.empty())
                    receiver->connect(s);

                receiver->start();
                std::cin.clear();

                //inputPort->startCapture();
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
    delete receiver;
    delete sender;
    delete card;


    clean_up();
}




