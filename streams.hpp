#pragma once
#include "input_output.hpp"
#include "ndi_api.hpp"
#include "console_control.hpp"

namespace ndi_deck {
	enum displayResoltion { HD, UHD };

	class Stream {
	protected:
		int stream_id;
		int type;

	public:
		virtual void start_stream() = 0; // starts streams
		virtual void stop_stream() = 0;
		virtual void init() = 0; // initialize stream NDI init and Deck Init

	};

	class OutputStream  : public Stream {
	private:
		BMDDisplayMode kInitialDisplayMode;
		BMDPixelFormat kInitialPixelFormat;
		bool kWaitForReferenceToLock;
		bool exitFlag;
		int kOutputVideoPreroll;
		int kVideoDispatcherThreadCount;
		com_ptr<IDeckLink>					deckLink;
		com_ptr<IDeckLink>					kDeckLink;
		com_ptr<DeckLinkOutputDevice>		deckLinkOutput;
		com_ptr<DeckLinkOutputDevice>		deckLinkOutput2;

		DispatchQueue 						videoDispatchQueue_s;
		NDI_Recv* receiver;
		FormatDescription formatDesc, currentFormatDesc;
		HRESULT result;
		DispatchQueue printDispatchQueue_s;
		const BMDAudioSampleType	kAudioSampleType;
	

	public:
		OutputStream(com_ptr<IDeckLink>& device, int res, int id=0);
		OutputStream(com_ptr<IDeckLink>& device,com_ptr<IDeckLink>&dev2, int res, int id = 0);
		void init() override;
		void start_stream() override;
		void stop_stream() override;
	};

	class StreamManager {
		std::vector<Stream *> streams;
		std::vector<com_ptr<IDeckLink>> unused_devices;
		std::vector<com_ptr<IDeckLink>> used_devices;
		HRESULT result;
		
		com_ptr<IDeckLinkIterator>			deckLinkIterator;
		com_ptr<IDeckLink>					deckLink;

	public:
		StreamManager();

		Stream* create_input_stream(int resolution = displayResoltion::HD);
		OutputStream* create_output_stream(int resolution = displayResoltion::HD);
		void kill_stream(Stream* s);
		void kill_stream(int id);
		void kill_all_streams();

		~StreamManager();

	};
}