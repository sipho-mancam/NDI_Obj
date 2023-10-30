
#include "streams.hpp"

using namespace ndi_deck;

OutputStream::OutputStream(com_ptr<IDeckLink>& device, int res, int id)
	: Stream(),
	kInitialPixelFormat(bmdFormat10BitYUV),
	kWaitForReferenceToLock(true),
	exitFlag(false),
	kOutputVideoPreroll(1),
	kVideoDispatcherThreadCount(3),
	deckLink(device.get()),
	videoDispatchQueue_s(kVideoDispatcherThreadCount),
	receiver(nullptr),
	printDispatchQueue_s(1), 
	kAudioSampleType(bmdAudioSampleType32bitInteger)
{
	if (res == displayResoltion::HD)
		kInitialDisplayMode = bmdModeHD1080i50;
	else if (res == displayResoltion::UHD)
		kInitialDisplayMode = bmdMode4K2160p50;

	currentFormatDesc = {kInitialDisplayMode, false, kInitialPixelFormat};
	stream_id = id;
	init();

}


void OutputStream::init()
{
	// Initialize DeckLink
	com_ptr<IDeckLinkProfileAttributes>		deckLinkAttributes(IID_IDeckLinkProfileAttributes, deckLink);
	int64_t									duplexMode;
	int64_t									videoIOSupport;
	int64_t									maxAudioChannels;

	if (!deckLinkAttributes)
	{
		fprintf(stderr, "Could not obtain the IDeckLinkAttributes interface - result %08x\n", result);
		deckLink = nullptr;
		
	}

	// Check whether device is an active state
	if ((deckLinkAttributes->GetInt(BMDDeckLinkDuplex, &duplexMode) != S_OK) ||
		((BMDDuplexMode)duplexMode == bmdDuplexInactive))
	{
		deckLink = nullptr;
	
	}

	// Get the IO support for device
	if (deckLinkAttributes->GetInt(BMDDeckLinkVideoIOSupport, &videoIOSupport) == S_OK)
	{
		if (deckLinkAttributes->GetInt(BMDDeckLinkMaximumAudioChannels, &maxAudioChannels) != S_OK)
		{
			deckLink = nullptr;
		}	
		

		if (!deckLinkOutput && (((BMDVideoIOSupport)videoIOSupport & bmdDeviceSupportsPlayback) != 0))
		{
			int64_t minimumPrerollFrames;
			if (deckLinkAttributes->GetInt(BMDDeckLinkMinimumPrerollFrames, &minimumPrerollFrames) != S_OK)
			{
				fprintf(stderr, "Failed to get the minumum required number of pre-roll frames\n");
			}

			if (kOutputVideoPreroll < minimumPrerollFrames)
			{
				//dispatch_printf(printDispatchQueue_s, "Warning: Specified video output preroll size is smaller than the minimum supported size; Changing preroll size from %d to %d.\n", kOutputVideoPreroll, minimumPrerollFrames);
			}

			int prerollFrames = std::max((int)minimumPrerollFrames, kOutputVideoPreroll);

			try
			{
				deckLinkOutput = make_com_ptr<DeckLinkOutputDevice>(deckLink, prerollFrames);
			}
			catch (const std::exception& e)
			{
				fprintf(stderr, "%s\n", e.what());
				
			}

			dispatch_printf(printDispatchQueue_s, "Using output device: %s\n", getDeckLinkDisplayName(deckLink).c_str());
		}
	}

	if (!deckLinkOutput)
	{
		fprintf(stderr, "Unable to find both active input and output devices\n");
		//return E_FAIL;
	}

	// Initialize NDI
	receiver = new NDI_Recv(&exitFlag, stream_id);

	TCHAR compName[MAX_COMPUTERNAME_LENGTH + 3];
	DWORD size = MAX_COMPUTERNAME_LENGTH + 3;

	GetComputerName(compName, &size);
	std::string connection_string(compName);
	connection_string += " (VizEngine-0)";

	receiver->connect(connection_string);
	// Register Input Callbacks
	receiver->onVideoInputArrived([&](std::shared_ptr<LoopThroughVideoFrame> videoFrame) { videoDispatchQueue_s.dispatch(processVideo, videoFrame, deckLinkOutput); });
	// Register output callbacks
	deckLinkOutput->onScheduledFrameCompleted([&](std::shared_ptr<LoopThroughVideoFrame> videoFrame) { updateCompletedFrameLatency(videoFrame, std::ref(printDispatchQueue_s)); });

	
}

void OutputStream::start_stream()
{
	std::mutex formatDescMutex;

	if (kWaitForReferenceToLock)
		dispatch_printf(printDispatchQueue_s, "Waiting for reference to lock...\n");

	if (!deckLinkOutput->startPlayback(currentFormatDesc.displayMode, currentFormatDesc.is3D, currentFormatDesc.pixelFormat, kAudioSampleType, 1, kWaitForReferenceToLock))
	{
		std::lock_guard<std::mutex> lock(formatDescMutex);
		if (!g_loopThroughSessionNotifier.isNotified() && formatDesc == currentFormatDesc)
		{
			fprintf(stderr, "Unable to enable output on the selected device\n");
		}
	}

	printReferenceStatus(deckLinkOutput, printDispatchQueue_s);
	dispatch_printf(printDispatchQueue_s, "Starting stream .... \n");

	// start NDI input
	receiver->start();
}

void OutputStream::stop_stream()
{
	// stop DeckLink Output
	deckLinkOutput->stopPlayback();
	printOutputSummary(printDispatchQueue_s);

	// stop ndi receiver
	receiver->stop();
}



StreamManager::StreamManager()
{
	result = GetDeckLinkIterator(deckLinkIterator.releaseAndGetAddressOf());
	if (result != S_OK)
		return;
	while (deckLinkIterator->Next(deckLink.releaseAndGetAddressOf()) == S_OK)
	{
		unused_devices.push_back(deckLink);
	}
}


Stream* StreamManager::create_input_stream(int resolution)
{
	return nullptr;
}


OutputStream* StreamManager::create_output_stream(int resolution)
{
	try {

		OutputStream* out_stream = new OutputStream(unused_devices[3], resolution);
		com_ptr<IDeckLink> dev = unused_devices[3];
		used_devices.push_back(dev);
		unused_devices.erase(unused_devices.begin()); // remove device from used to track the devices we still have left.

		streams.push_back(out_stream);
		std::cout << "Stream Created Successfully" << std::endl;
		return out_stream;

	}catch (std::exception& e) {
		std::cerr << "Unable to create any more streams, out of Decklink Ports" << std::endl;
		return nullptr;
	}
}

void StreamManager::kill_all_streams()
{
	for (Stream* stream : streams)
	{
		stream->stop_stream();
	}
}

StreamManager::~StreamManager()
{
	kill_all_streams();
}