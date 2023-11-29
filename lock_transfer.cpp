#include "lock_transfer.hpp"



ImplicitLock::ImplicitLock() :
	m_release(false)
{
	
}

void ImplicitLock::wait()
{
	std::unique_lock<std::mutex> lock(mtx); // obtain the lock
	cv.wait(lock);
	lock.unlock();
	cv.notify_one();
	//while (!m_release); // wait here until we cleared
	m_release = false; // reset the lock again
}

bool ImplicitLock::isLocked()
{
	{
		std::lock_guard<std::mutex> lk(mtx);
		return m_release;
	}
}

void ImplicitLock::release()
{
	// release the lock and notify waiters.
	{
		std::lock_guard<std::mutex> lk(mtx);
		m_release = true;
		cv.notify_one(); // let the thread know it's time to release
	}
	
}

void ImplicitLock::notify()
{
	//std::lock_guard<std::mutex> lock(mtx);
	m_release = true;
	cv.notify_one();
}

void ImplicitLock::reset()
{
	m_release = false;
}
