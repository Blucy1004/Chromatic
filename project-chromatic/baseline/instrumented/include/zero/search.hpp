#pragma once
#include "network.hpp"
#include <atomic>
#include <chrono>
#include <memory>
#include <random>
namespace zero {
struct SearchLimits {int simulations=800;int milliseconds=0;float cpuct=1.5f;float noise=0;float temperature=0;uint64_t seed=1;size_t max_nodes=100000;};
struct SearchResult {Move best{};bool has_move=false;bool terminal=false;int simulations=0;float value=0;std::vector<std::pair<Move,int>> visits;};
SearchResult search(const Board& board,const Network& network,const SearchLimits& limits,const std::atomic<bool>& stop);
}
