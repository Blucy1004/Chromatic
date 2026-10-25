#pragma once
#include "board.hpp"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace zero {
// Scores are centipawns from the root side-to-move perspective; mate values
// use MATE_SCORE - ply. A callback supplies advisory ordering values only.
inline constexpr int PVS_MATE_SCORE = 30000;
struct PvsLimits {
    int max_depth=6;
    uint64_t max_nodes=1000000;
    int milliseconds=0; // 0 means no wall-clock limit
    size_t tt_megabytes=16;
    int ordering_hint_min_depth=0;
    std::function<std::vector<float>(const Board&,const std::vector<Move>&)> ordering_hint{};
};
struct PvsResult {
    Move best{};
    bool has_move=false, terminal=false, completed=false;
    int score_cp=0, completed_depth=0;
    uint64_t nodes=0, qnodes=0, tt_hits=0, tt_cutoffs=0;
    std::vector<Move> pv;
    std::string stop_reason;
};
class PvsSearch {
public:
    explicit PvsSearch(size_t tt_megabytes=16);
    ~PvsSearch();
    PvsSearch(PvsSearch&&) noexcept;
    PvsSearch& operator=(PvsSearch&&) noexcept;
    PvsSearch(const PvsSearch&)=delete;
    PvsSearch& operator=(const PvsSearch&)=delete;
    void clear();
    PvsResult run(const Board& root,const PvsLimits& limits,const std::atomic<bool>& stop);
private:
    struct Impl;
    Impl* impl_;
};
}
