
#include "internal_sync.hpp"

Synchronizer::Synchronizer()
	: start_out(false), 
	monitor_started(false),
	delay(1),
	worker(nullptr)
{}

void Synchronizer::add_output(DeckLinkOutputPort* p)
{
	if (!monitor_started)
	{
		// you must add objects before starting the synchronizer
		p->synchronize(&start_out);
		outputs.push_back(p);
	}
	else {
		throw(std::exception("Can't add objects to synchronizer after sync is started."));
	}
}

void Synchronizer::remove_output(int c)
{
	DeckLinkOutputPort* port = outputs[c];
	port->synchronize(nullptr);

	outputs.erase(outputs.begin() + c);
}

void Synchronizer::start()
{
	// start the sync thread ... and monitor objects to alter
	if (!worker)
	{
		worker = new std::thread(&Synchronizer::sync_objects, this);
		monitor_started = true;
	}
}

void Synchronizer::set_delay(int c)
{
	delay = c;
}

int Synchronizer::get_delay()
{
	return delay;
}

bool Synchronizer::is_monitor_started()
{
	return monitor_started;
}

bool Synchronizer::is_synced()
{
	return start_out;
}

void Synchronizer::sync_objects()
{
	bool _temp_flag = false;
	while (monitor_started)
	{
		for (DeckLinkOutputPort* p : outputs)
		{
			std::queue<IDeckLinkVideoFrame*>* q = p->get_output_q();
			if (q)
			{
				if (q->size() >= delay)
					_temp_flag = true;
				else {
					_temp_flag = false;
					break;
				}	
			}
		}
		start_out = _temp_flag;
		if (start_out)
			break;
	}
	//worker->join();
}

Synchronizer::~Synchronizer()
{
	monitor_started = false; // stop the monitor thread ...

	std::this_thread::sleep_for(std::chrono::milliseconds(5));

	if (!start_out)
	{
		for (DeckLinkOutputPort* p : outputs)
		{
			p->synchronize(nullptr); // remove sync from all objects ... so they can release at will.
		}
	}

	if (worker)
	{
		worker->join();
		delete worker;
		worker = nullptr;
	}
		
}



