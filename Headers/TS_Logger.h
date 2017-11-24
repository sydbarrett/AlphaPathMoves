/*
This software implements a thread safe logger.
This and is provied "AS IS" without any warranty, please see disclaimer below.

##################################################################

License & disclaimer.

Copyright Hossam Isack 	<isack.hossam@gmal.com>

This software and its modifications can be used and distributed for
research purposes only.Publications resulting from use of this code
must cite publications according to the rules given above.Only
Hossam Isack has the right to redistribute this code, unless expressed
permission is given otherwise.Commercial use of this code, any of
its parts, or its modifications is not permited.The copyright notices
must not be removed in case of any modifications.This Licence
commences on the date it is electronically or physically delivered
to you and continues in effect unless you fail to comply with any of
the terms of the License and fail to cure such breach within 30 days
of becoming aware of the breach, in which case the Licence automatically
terminates.This Licence is governed by the laws of Canada and all
disputes arising from or relating to this Licence must be brought
in Toronto, Ontario.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES(INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

##################################################################
*/

#ifndef __TS_LOGGER_H
#define __TS_LOGGER_H
#include <mutex>
#include <fstream> 
#include <sstream>
#include <thread>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdint>
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
	TS_Logger& operator<<(const double&);
	TS_Logger& operator<<(const float&);
	TS_Logger& operator<<(const int64_t &);
	TS_Logger& operator<<(const uint64_t &);
	TS_Logger& operator<<(const int32_t&);
	TS_Logger& operator<<(const uint32_t&);
	TS_Logger& operator<<(const int16_t&);
	TS_Logger& operator<<(const uint16_t&);
	TS_Logger& operator<<(const int8_t&);
	TS_Logger& operator<<(const uint8_t&);
	
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