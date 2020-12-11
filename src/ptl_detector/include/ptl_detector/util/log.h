#pragma once

#include <cstdarg>
#include <string>

#include "glog/logging.h"
#include "glog/raw_logging.h"

#define ADEBUG VLOG(4) << "[DEBUG] "
#define AINFO ALOG(INFO)
#define AWARN ALOG(WARN)
#define AERROR ALOG(ERROR)
#define AFATAL ALOG(FATAL)

#ifndef ALOG_STREAM
#define ALOG_STREAM(log_severity) ALOG_STREAM_##log_severity
#endif

#ifndef ALOG
#define ALOG(log_severity) ALOG_STREAM(log_severity)
#endif

#define ALOG_STREAM_INFO \
    google::LogMessage(__FILE__, __LINE__, google::INFO).stream()

#define ALOG_STREAM_WARN \
    google::LogMessage(__FILE__, __LINE__, google::WARNING).stream()

#define ALOG_STREAM_ERROR \
    google::LogMessage(__FILE__, __LINE__, google::ERROR).stream()

#define ALOG_STREAM_FATAL \
    google::LogMessage(__FILE__, __LINE__, google::FATAL).stream()

#define AINFO_IF(cond) ALOG_IF(INFO, cond)
#define AWARN_IF(cond) ALOG_IF(WARN, cond)
#define AERROR_IF(cond) ALOG_IF(ERROR, cond)
#define AFATAL_IF(cond) ALOG_IF(FATAL, cond)
#define ALOG_IF(severity, cond) \
    !(cond) ? (void)0 : google::LogMessageVoidify() & ALOG(severity)

#define ACHECK(cond) CHECK(cond)

#define AINFO_EVERY(freq) LOG_EVERY_N(INFO, freq)
#define AWARN_EVERY(freq) LOG_EVERY_N(WARNING, freq)
#define AERROR_EVERY(freq) LOG_EVERY_N(ERROR, freq)

#if !defined(RETURN_IF_NULL)
#define RETURN_IF_NULL(ptr)              \
    if (ptr == nullptr)                  \
    {                                    \
        AWARN << #ptr << " is nullptr."; \
        return;                          \
    }
#endif

#if !defined(RETURN_VAL_IF_NULL)
#define RETURN_VAL_IF_NULL(ptr, val)     \
    if (ptr == nullptr)                  \
    {                                    \
        AWARN << #ptr << " is nullptr."; \
        return val;                      \
    }
#endif

#if !defined(RETURN_IF)
#define RETURN_IF(condition)               \
    if (condition)                         \
    {                                      \
        AWARN << #condition << " is met."; \
        return;                            \
    }
#endif

#if !defined(RETURN_VAL_IF)
#define RETURN_VAL_IF(condition, val)      \
    if (condition)                         \
    {                                      \
        AWARN << #condition << " is met."; \
        return val;                        \
    }
#endif

#if !defined(_RETURN_VAL_IF_NULL2__)
#define _RETURN_VAL_IF_NULL2__
#define RETURN_VAL_IF_NULL2(ptr, val) \
    if (ptr == nullptr)               \
    {                                 \
        return (val);                 \
    }
#endif

#if !defined(_RETURN_VAL_IF2__)
#define _RETURN_VAL_IF2__
#define RETURN_VAL_IF2(condition, val) \
    if (condition)                     \
    {                                  \
        return (val);                  \
    }
#endif

#if !defined(_RETURN_IF2__)
#define _RETURN_IF2__
#define RETURN_IF2(condition) \
    if (condition)            \
    {                         \
        return;               \
    }
#endif
