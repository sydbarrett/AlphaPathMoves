#ifndef __TS_LOGGER_H
#define __TS_LOGGER_H
#include <mutex>
#include <fstream> 
#include <sstream>
#include <thread>
#include <iostream>
#include <iomanip>
#include <vector>
#include <initializer_list>
#undef ERROR
using namespace std;

enum class LogType : std::uint32_t { INFO_LEVEL0 = 1, INFO_LEVEL1 = 2, INFO_LEVEL2 = 3, INFO_LEVEL3 = 4, INFO_LEVEL4 = 5, DEBUG = 6, ERROR = 7, WARNING = 8 };
class TS_Logger
{

private:
	static bool process_event;
	static std::ofstream log_file;
	static std::uint32_t LOGGED_TYPES;
	TS_Logger() {};
	static bool isEventLogged(const LogType &);
public:
	static TS_Logger in_bgn_log;
	~TS_Logger()
	{
		cout_mtx.lock();
		log_file.close();
		cout_mtx.unlock();
	};
	TS_Logger(const TS_Logger&) = delete;
	TS_Logger(TS_Logger&&) = delete;
	TS_Logger& operator=(TS_Logger&& inLine) = delete;
	TS_Logger& operator=(const TS_Logger& inLine) = delete;
	class EndLog {};
	static std::mutex cout_mtx;
	TS_Logger& operator<<(const LogType&);
	TS_Logger& operator<<(const string&);
	TS_Logger& operator<<(const char&);
	TS_Logger& operator<<(const double&);
	TS_Logger& operator<<(const long long &);
	TS_Logger& operator<<(const unsigned int&);
	TS_Logger& operator<<(const unsigned long int&);
	TS_Logger& operator<<(const int&);
	TS_Logger& operator<<(const long int&);
	TS_Logger& operator<<(const EndLog&);
	static void LogEvent(LogType);
	static void LogEvents(std::initializer_list<LogType> list);
	static void LogAllEvents();
	static void UnlogEvent(LogType);
	static void UnlogAllEvents();
	static void UnlogEvents(std::initializer_list<LogType> list);

};
static TS_Logger::EndLog end_log;
static TS_Logger & bgn_log = TS_Logger::in_bgn_log;
#endif