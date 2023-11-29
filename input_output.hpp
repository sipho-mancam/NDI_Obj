#pragma once


#include <chrono>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <thread>

#include "DeckLinkOutputDevice.h"
#include "DeckLinkInputDevice.h"
#include "DispatchQueue.hpp"
#include "SampleQueue.h"
#include "LatencyStatistics.h"
#include "ReferenceTime.h"
#include "DeckLinkAPI.h"
#include "com_ptr.h"
#include "platform.h"
#include "ndi_api.hpp"

static const bool					kPrintRollingAverage = true;		// If true, display latency as rolling average, if false print latency for each frame
static const int					kRollingAverageSampleCount = 300;		// Number of samples for calculating rolling average of latency
static const long					kRollingAverageUpdateRateMs = 2000;		// Print rolling average every 2 seconds

static const double					kProcessingAdditionalTimeMean = 5.0;		// Mean additional time injected into video processing thread (ms)
static const double					kProcessingAdditionalTimeStdDev = 0.1;		// Standard deviation of time injected into video processing thread (ms)


// Output frame completion result pair = { Completion result string, frame output boolean}
static const std::map<BMDOutputFrameCompletionResult, std::pair<const char*, bool>> kOutputCompletionResults
{
	{ bmdOutputFrameCompleted,		std::make_pair("completed",			true) },
	{ bmdOutputFrameDisplayedLate,	std::make_pair("displayed late",	true) },
	{ bmdOutputFrameDropped,		std::make_pair("dropped",			false) },
	{ bmdOutputFrameFlushed,		std::make_pair("flushed",			false) },
};

// List of known pixel formats and their matching display names
static const std::map<BMDPixelFormat, const char*> kPixelFormats =
{
	{ bmdFormat8BitYUV,		"8-bit YUV" },
	{ bmdFormat10BitYUV,	"10-bit YUV" },
	{ bmdFormat8BitARGB,	"8-bit ARGB" },
	{ bmdFormat8BitBGRA,	"8-bit BGRA" },
	{ bmdFormat10BitRGB,	"10-bit RGB" },
	{ bmdFormat12BitRGB,	"12-bit RGB" },
	{ bmdFormat12BitRGBLE,	"12-bit RGBLE" },
	{ bmdFormat10BitRGBXLE,	"10-bit RGBXLE" },
	{ bmdFormat10BitRGBX,	"10-bit RGBX" },
};

struct ThreadNotifier
{
	std::mutex mutex;
	std::condition_variable condition;

	ThreadNotifier() :
		m_notified(false)
	{ }

	void reset()
	{
		std::lock_guard<std::mutex> lock(mutex);
		m_notified = false;
	}

	void notify()
	{
		std::lock_guard<std::mutex> lock(mutex);
		m_notified = true;
		condition.notify_all();
	}

	bool isNotified()
	{
		std::lock_guard<std::mutex> lock(mutex);
		return m_notified;
	}

	bool isNotifiedLocked()
	{
		return m_notified;
	}

private:
	bool m_notified;
};


//uint32_t														g_audioChannelCount = kDefaultAudioChannelCount;

static LatencyStatistics												g_videoInputLatencyStatistics(kRollingAverageSampleCount);
static LatencyStatistics												g_videoProcessingLatencyStatistics(kRollingAverageSampleCount);
static LatencyStatistics												g_videoOutputLatencyStatistics(kRollingAverageSampleCount);
static LatencyStatistics												g_audioProcessingLatencyStatistics(kRollingAverageSampleCount);

static std::map<BMDOutputFrameCompletionResult, int>					g_frameCompletionResultCount;
static int 															g_outputFrameCount = 0;
static int																g_droppedOnCaptureFrameCount = 0;

static std::default_random_engine 										g_randomEngine;
static std::normal_distribution<double> 								g_sleepDistribution(kProcessingAdditionalTimeMean, kProcessingAdditionalTimeStdDev);

static ThreadNotifier													g_printRollingAverageNotifier;
static ThreadNotifier													g_loopThroughSessionNotifier;

struct FormatDescription
{
	BMDDisplayMode displayMode;
	bool is3D;
	BMDPixelFormat pixelFormat;
};

bool operator==(const FormatDescription& desc1, const FormatDescription& desc2);
bool operator!=(const FormatDescription& desc1, const FormatDescription& desc2);


void processVideo(std::shared_ptr<LoopThroughVideoFrame>& videoFrame, com_ptr<DeckLinkOutputDevice>& deckLinkOutput);
void processVideo2(std::shared_ptr<LoopThroughVideoFrame>& videoFrame, std::shared_ptr<LoopThroughVideoFrame>& kSig, com_ptr<DeckLinkOutputDevice>& deckLinkOutput, com_ptr<DeckLinkOutputDevice>& deckLinkOutput2);
std::string getDeckLinkDisplayName(com_ptr<IDeckLink> deckLink);
void printDroppedCaptureFrame(BMDTimeValue streamTime, BMDTimeValue frameDuration, DispatchQueue& printDispatchQueue);
void printOutputCompletionResult(std::shared_ptr<LoopThroughVideoFrame> completedFrame, DispatchQueue& printDispatchQueue);
void updateCompletedFrameLatency(std::shared_ptr<LoopThroughVideoFrame> completedFrame, DispatchQueue& printDispatchQueue);
void printRollingAverage(DispatchQueue& printDispatchQueue);
void printOutputSummary(DispatchQueue& printDispatchQueue);
void printReferenceStatus(com_ptr<DeckLinkOutputDevice>& deckLinkOutput, DispatchQueue& printDispatchQueue);


template<typename... Args>
void dispatch_printf(DispatchQueue& dispatchQueue, const char* format, Args... args);