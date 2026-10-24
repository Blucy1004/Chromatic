#pragma once
#include <chrono>
#include <cstdint>
namespace zero::profile {
inline thread_local bool active=false;
inline thread_local uint64_t neural_ns=0, neural_calls=0, movegen_ns=0, feature_ns=0;
inline void reset(){neural_ns=neural_calls=movegen_ns=feature_ns=0;}
struct Timer {
    uint64_t* counter;
    std::chrono::steady_clock::time_point start;
    explicit Timer(uint64_t& target):counter(active?&target:nullptr){if(counter)start=std::chrono::steady_clock::now();}
    ~Timer(){if(counter)*counter+=uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count());}
};
}
