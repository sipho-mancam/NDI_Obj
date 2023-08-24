#pragma once

#include <iostream>
#include <vector>
#include <queue>
#include "decklink_api.hpp"
#include <thread>
#include <chrono>

class Synchronizer {
private:
	bool start_out, monitor_started; // this is the flag to signal output ...
	std::vector<DeckLinkOutputPort*> outputs;
	std::thread* worker;
	int delay;

	void sync_objects(); // iterate all objects and add a flag.

public:
	Synchronizer();

	void add_output(DeckLinkOutputPort* p);
	void remove_output(int c);
	void start();
	
	void set_delay(int d);
	int get_delay();

	bool is_monitor_started();
	bool is_synced();

	~Synchronizer();
};
