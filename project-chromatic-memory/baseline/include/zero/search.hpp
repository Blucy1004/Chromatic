#pragma once
#include "network.hpp"
#include <atomic>
#include <chrono>
#include <memory>
#include <random>
namespace zero {
struct SearchLimits {
    int simulations=800,milliseconds=0;
    float cpuct=1.5f,noise=0,temperature=0,fpu_reduction=0;
    uint64_t seed=1;
    size_t max_nodes=100000,max_tree_bytes=64*1024*1024;
    bool tactical_safety=false;
    int proof_nodes=0,proof_plies=1;
};
struct SearchResult {
    Move best{};bool has_move=false,terminal=false;
    int simulations=0,neural_evals=0,max_depth=0;
    size_t arena_nodes=0,memory_bytes=0;
    float value=0;
    int proof_nodes=0,proof_plies=0;
    bool mate_proven=false;
    std::vector<std::pair<Move,int>> visits;
    std::vector<float> q_values,priors;
};
SearchResult search(const Board& board,const Network& network,const SearchLimits& limits,const std::atomic<bool>& stop);
}
