#pragma once
#include "board.hpp"
#include <array>
#include <chrono>
#include <cstdint>
namespace zero {
struct ResearchMetrics {
    std::array<uint64_t,513> leaf_depth{};
    uint64_t node_creations=0,edge_expansions=0,stored_edges=0;
    uint64_t adaptive_calls=0,verification_calls=0,admission_calls=0,admission_nodes=0;
    uint64_t counterexample_calls=0,counterexample_replies=0,quiescence_nodes=0,forcing_checks=0,material_calls=0;
    uint64_t parallel_init_ns=0,parallel_wait_handoff_ns=0,parallel_join_ns=0,root_nn_ns=0;
    uint64_t root_ns=0,mcts_ns=0,nn_mcts_ns=0,adaptive_ns=0,verification_ns=0;
    uint64_t admission_ns=0,counterexample_exclusive_ns=0,quiescence_ns=0,nn_verification_ns=0,forcing_checks_ns=0,material_ns=0,proof_ns=0,prune_ns=0;
    uint64_t algorithm_ns=0,call_ns=0,selection_ties=0;
    double minimum_positive_gap=1e100;
    std::vector<Move> pv;
};
class ResearchTimer {
    uint64_t* target_;
    std::chrono::steady_clock::time_point start_;
public:
    explicit ResearchTimer(uint64_t* target):target_(target){if(target_)start_=std::chrono::steady_clock::now();}
    ~ResearchTimer(){if(target_)*target_+=uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start_).count());}
};
}
