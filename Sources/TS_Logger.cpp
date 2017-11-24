#include "TS_Logger.h"

std::mutex TS_Logger::cout_mtx;
std::ofstream TS_Logger::log_file = std::ofstream("ts_log.txt", std::ios::out);
std::uint32_t TS_Logger::LOGGED_TYPES = 7; //deafult := log all 3 events events
bool TS_Logger::process_event = false;
TS_Logger TS_Logger::in_bgn_log;

bool TS_Logger::isEventLogged(const LogType & in_logtype)
{
	return (LOGGED_TYPES & static_cast<std::uint32_t>(in_logtype)) != 0;
}

TS_Logger& TS_Logger::operator<<(const LogType& in_logtype)
{
	cout_mtx.lock();
	process_event = true;
	if (!isEventLogged(in_logtype))
		process_event = false;
	std::stringstream thread_id, printed_msg;
	thread_id << std::setw(8) << std::setfill('0') << std::this_thread::get_id();
	switch (in_logtype)
	{
	case  LogType::DEBUG:
		printed_msg << "[" << thread_id.str() << "] DEBUG: ";
		break;
	case LogType::ERROR:
		printed_msg << "[" << thread_id.str() << "] ERROR: ";
		break;
	default:
		printed_msg << "[" << thread_id.str() << "] INFO_LEVEL" + std::to_string(static_cast<std::uint32_t>(in_logtype) - 1) + ": ";
		break;
	}
	if (process_event)
	{
		std::cout << printed_msg.str() << std::flush;
		log_file << printed_msg.str();
		log_file.flush();
	}
	return *this;
}

TS_Logger& TS_Logger::operator<<(const string& msg)
{
	if (process_event)
	{
		std::cout << msg << std::flush;
		log_file << msg;
		log_file.flush();
	}

	return *this;
}

TS_Logger& TS_Logger::operator<<(const int8_t& msg)
{
	if (process_event)
	{
		std::cout << std::to_string(msg) << std::flush;
		log_file << std::to_string(msg);
		log_file.flush();
	}
	return *this;
}
TS_Logger& TS_Logger::operator<<(const uint8_t& msg)
{
	if (process_event)
	{
		std::cout << std::to_string(msg) << std::flush;
		log_file << std::to_string(msg);
		log_file.flush();
	}
	return *this;
}

TS_Logger& TS_Logger::operator<<(const int16_t& msg)
{
	if (process_event)
	{
		std::cout << std::to_string(msg) << std::flush;
		log_file << std::to_string(msg);
		log_file.flush();
	}
	return *this;
}
TS_Logger& TS_Logger::operator<<(const uint16_t& msg)
{
	if (process_event)
	{
		std::cout << std::to_string(msg) << std::flush;
		log_file << std::to_string(msg);
		log_file.flush();
	}
	return *this;
}
TS_Logger& TS_Logger::operator<<(const int32_t& msg)
{
	if (process_event)
	{
		std::cout << std::to_string(msg) << std::flush;
		log_file << std::to_string(msg);
		log_file.flush();
	}
	return *this;
}
TS_Logger & TS_Logger::operator<<(const uint32_t & msg)
{
	if (process_event)
	{
		std::cout << std::to_string(msg) << std::flush;
		log_file << std::to_string(msg);
		log_file.flush();
	}
	return *this;
}
TS_Logger & TS_Logger::operator<<(const int64_t &msg)
{
	if (process_event)
	{
		std::cout << std::to_string(msg) << std::flush;
		log_file << std::to_string(msg);
		log_file.flush();
	}
	return *this;
}
TS_Logger & TS_Logger::operator<<(const uint64_t &msg)
{
	if (process_event)
	{
		std::cout << std::to_string(msg) << std::flush;
		log_file << std::to_string(msg);
		log_file.flush();
	}
	return *this;
}
TS_Logger& TS_Logger::operator<<(const double& msg)
{
	if (process_event)
	{
		std::cout << std::to_string(msg) << std::flush;
		log_file << std::to_string(msg);
		log_file.flush();
	}
	return *this;
}
TS_Logger& TS_Logger::operator<<(const float& msg)
{
	if (process_event)
	{
		std::cout << std::to_string(msg) << std::flush;
		log_file << std::to_string(msg);
		log_file.flush();
	}
	return *this;
}

TS_Logger& TS_Logger::operator<<(const EndLog& obj)
{
	cout_mtx.unlock();
	return *this;
}
void TS_Logger::LogEvent(LogType in_logtype)
{
	LOGGED_TYPES |= (1 << (static_cast<std::uint32_t>(in_logtype) - 1));

	if (isEventLogged(LogType::INFO_LEVEL4))
		LogEvent(LogType::INFO_LEVEL3);
	if (isEventLogged(LogType::INFO_LEVEL3))
		LogEvent(LogType::INFO_LEVEL2);
	if (isEventLogged(LogType::INFO_LEVEL2))
		LogEvent(LogType::INFO_LEVEL1);
	if (isEventLogged(LogType::INFO_LEVEL1))
		LogEvent(LogType::INFO_LEVEL0);
}
void TS_Logger::LogEvents(std::initializer_list<LogType> list)
{
	for (auto &log_event : list)
		LogEvent(log_event);
}
void TS_Logger::LogAllEvents()
{
	LOGGED_TYPES = 0;
	LOGGED_TYPES |= (1 << (static_cast<std::uint32_t>(LogType::INFO_LEVEL0) - 1));
	LOGGED_TYPES |= (1 << (static_cast<std::uint32_t>(LogType::INFO_LEVEL1) - 1));
	LOGGED_TYPES |= (1 << (static_cast<std::uint32_t>(LogType::INFO_LEVEL2) - 1));
	LOGGED_TYPES |= (1 << (static_cast<std::uint32_t>(LogType::INFO_LEVEL3) - 1));
	LOGGED_TYPES |= (1 << (static_cast<std::uint32_t>(LogType::INFO_LEVEL4) - 1));
	LOGGED_TYPES |= (1 << (static_cast<std::uint32_t>(LogType::DEBUG) - 1));
	LOGGED_TYPES |= (1 << (static_cast<std::uint32_t>(LogType::ERROR) - 1));
}
void TS_Logger::UnlogEvent(LogType in_logtype)
{
	LOGGED_TYPES &= ~(1 << (static_cast<std::uint32_t>(in_logtype) - 1));
	if (isEventLogged(LogType::INFO_LEVEL3))
		UnlogEvent(LogType::INFO_LEVEL4);
	if (isEventLogged(LogType::INFO_LEVEL2))
		UnlogEvent(LogType::INFO_LEVEL3);
	if (isEventLogged(LogType::INFO_LEVEL1))
		UnlogEvent(LogType::INFO_LEVEL2);
	if (isEventLogged(LogType::INFO_LEVEL0))
		UnlogEvent(LogType::INFO_LEVEL1);
}
void TS_Logger::UnlogAllEvents()
{
	LOGGED_TYPES = 0;
}

void TS_Logger::UnlogEvents(std::initializer_list<LogType> list)
{
	for (auto &log_event : list)
		UnlogEvent(log_event);
}
