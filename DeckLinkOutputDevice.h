/* -LICENSE-START-
** Copyright (c) 2019 Blackmagic Design
**  
** Permission is hereby granted, free of charge, to any person or organization 
** obtaining a copy of the software and accompanying documentation (the 
** "Software") to use, reproduce, display, distribute, sub-license, execute, 
** and transmit the Software, and to prepare derivative works of the Software, 
** and to permit third-parties to whom the Software is furnished to do so, in 
** accordance with:
** 
** (1) if the Software is obtained from Blackmagic Design, the End User License 
** Agreement for the Software Development Kit (“EULA”) available at 
** https://www.blackmagicdesign.com/EULA/DeckLinkSDK; or
** 
** (2) if the Software is obtained from any third party, such licensing terms 
** as notified by that third party,
** 
** and all subject to the following:
** 
** (3) the copyright notices in the Software and this entire statement, 
** including the above license grant, this restriction and the following 
** disclaimer, must be included in all copies of the Software, in whole or in 
** part, and all derivative works of the Software, unless such copies or 
** derivative works are solely in the form of machine-executable object code 
** generated by a source language processor.
** 
** (4) THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
** OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
** FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
** SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
** FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
** ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
** DEALINGS IN THE SOFTWARE.
** 
** A copy of the Software is available free of charge at 
** https://www.blackmagicdesign.com/desktopvideo_sdk under the EULA.
** 
** -LICENSE-END-
*/

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <thread>

#include "DeckLinkAPI.h"
#include "LoopThroughVideoFrame.h"
#include "SampleQueue.h"
#include "platform.h"
#include "com_ptr.h"

class DeckLinkOutputDevice : public IDeckLinkVideoOutputCallback, public IDeckLinkAudioOutputCallback
{
	enum class PlaybackState { Idle, Starting, Prerolling, Running, Stopping, Stopped };

	using ScheduledFrameCompletedCallback	= std::function<void(std::shared_ptr<LoopThroughVideoFrame>)>;	
	using ScheduledFramesList				= std::list<std::shared_ptr<LoopThroughVideoFrame>>;

public:
	DeckLinkOutputDevice(com_ptr<IDeckLink>& deckLink, int videoPrerollSize);
	virtual ~DeckLinkOutputDevice() = default;

	// IUnknown interface
	HRESULT						STDMETHODCALLTYPE QueryInterface(REFIID iid, LPVOID *ppv) override;
	ULONG						STDMETHODCALLTYPE AddRef() override;
	ULONG						STDMETHODCALLTYPE Release() override;

	// IDeckLinkVideoOutputCallback interface
	HRESULT						STDMETHODCALLTYPE ScheduledFrameCompleted(IDeckLinkVideoFrame* completedFrame, BMDOutputFrameCompletionResult result) override;
	HRESULT						STDMETHODCALLTYPE ScheduledPlaybackHasStopped() override;

	// IDeckLinkAudioOutputCallback interface
	HRESULT						STDMETHODCALLTYPE RenderAudioSamples(dlbool_t preroll) override;

	// Other methods
	bool						startPlayback(BMDDisplayMode displayMode, bool enable3D, BMDPixelFormat pixelFormat, BMDAudioSampleType audioSampleType, uint32_t audioChannelCount, bool requireReferenceLocked);
	void						stopPlayback(void);

	void						cancelWaitForReference();

	BMDTimeScale				getFrameTimescale(void) const { return m_frameTimescale; }
	com_ptr<IDeckLinkOutput>	getDeckLinkOutput(void) const { return m_deckLinkOutput; }
	bool						getReferenceSignalMode(BMDDisplayMode* mode);
	bool						isPlaybackActive(void);
	void						scheduleVideoFrame(std::shared_ptr<LoopThroughVideoFrame> videoFrame) { m_outputVideoFrameQueue.pushSample(videoFrame); }
	void						onScheduledFrameCompleted(const ScheduledFrameCompletedCallback& callback) { m_scheduledFrameCompletedCallback = callback; }


private:
	std::atomic<ULONG>										m_refCount;
	PlaybackState											m_state;
	//
	com_ptr<IDeckLink>										m_deckLink;
	com_ptr<IDeckLinkOutput>								m_deckLinkOutput;
	//
	SampleQueue<std::shared_ptr<LoopThroughVideoFrame>>		m_outputVideoFrameQueue;
	ScheduledFramesList										m_scheduledFramesList;
	//
	uint32_t												m_videoPrerollSize;
	//
	BMDTimeValue											m_frameDuration;
	BMDTimeScale											m_frameTimescale;
	//
	bool													m_seenFirstVideoFrame;
	BMDTimeValue											m_startPlaybackTime;
	//
	std::mutex												m_mutex;
	std::condition_variable									m_playbackStoppedCondition;
	//
	std::thread												m_scheduleVideoFramesThread;
	//
	ScheduledFrameCompletedCallback							m_scheduledFrameCompletedCallback;
	//

	// Private methods
	void		scheduleVideoFramesThread(void);
	//void		scheduleAudioPacketsThread(void);
	bool		waitForReferenceSignalToLock();
	void 		checkEndOfPreroll(void);

};