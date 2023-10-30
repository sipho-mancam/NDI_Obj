#pragma once
#include "input_output.hpp"
#include "ndi_api.hpp"

namespace ndi_deck {
	enum displayResoltion { HD, UHD };

	class Stream {
	protected:
		int stream_id;
		int type;
	
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
		com_ptr<DeckLinkOutputDevice>		deckLinkOutput;
		DispatchQueue 						videoDispatchQueue_s;
		NDI_Recv* receiver;
		FormatDescription formatDesc, currentFormatDesc;
		HRESULT result;
		DispatchQueue printDispatchQueue_s;
		const BMDAudioSampleType	kAudioSampleType;
	

	public:
		OutputStream(com_ptr<IDeckLink>& device, displayResoltion res, int id=0);
		void init() override;
		void start_stream() override;
		void stop_stream() override;
	};

	class StreamManager {

	};
}