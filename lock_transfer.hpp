#pragma once

#include <condition_variable>
#include <mutex>


class ImplicitLock {
public:
	ImplicitLock();
	void wait(); // wait until triggered to go. set the variable and wait.
	void release(); // release the lock.
	void reset(); // reset the variable to unlocked.
	bool isLocked(); // check the conditional variable
	void notify();
	

private:
	std::mutex mtx;
	bool m_release;
	std::condition_variable cv;
};

//
//const static ImplicitLock lock_transfer;