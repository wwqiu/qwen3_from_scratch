#pragma once
#include <cstdarg>

#define LOG_LEVEL_ERROR 0
#define LOG_LEVEL_WARNING 1
#define LOG_LEVEL_INFO 2
#define LOG_LEVEL_DEBUG 3

#define LOG_INFO(fmt, ...)  LOGF(LOG_LEVEL_INFO, fmt, ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) LOGF(LOG_LEVEL_DEBUG, fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) LOGF(LOG_LEVEL_ERROR, fmt, ##__VA_ARGS__)

void LOGF(int level, const char* fmt, ...) {
    uint32_t LOG_LEVEL = getenv("LOG_LEVEL") ? atoi(getenv("LOG_LEVEL")) : LOG_LEVEL_INFO;
    if (level > LOG_LEVEL) return;
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}