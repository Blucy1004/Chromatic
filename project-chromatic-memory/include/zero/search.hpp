#pragma once
#include "network.hpp"
#include <atomic>
#include <chrono>
#include <memory>
#include <random>
#include <array>
#include <cstdint>
#include <string>
namespace zero {
// Depth is measured in plies below this search root. Each completed MCTS
// simulation contributes exactly one endpoint, including cached terminal leaves.
// Allocation failures are aborted attempts, not backed-up MCTS simulations.
// Root initialization and mate proof updates are excluded from histograms.
inline constexpr size_t TELEMETRY_DEPTH_BINS=513;
using DepthHistogram=std::array<uint64_t,TELEMETRY_DEPTH_BINS>;
struct DepthSummary {
    uint64_t samples=0,sum_depth=0;
    int max_depth=0;
    double mean_depth=0,median_depth=0;
};
struct TelemetryCounters {
    uint64_t node_creations=0; // retained nodes, excluding the root
    uint64_t legal_expansions=0,root_legal_expansions=0; // successful edge-list expansions
    uint64_t child_fail_node_limit=0,child_fail_byte_capacity=0,expansion_fail_byte_capacity=0;
    uint64_t neural_evaluations=0,non_retained_neural_evaluations=0,resident_unexpanded_repeat=0;
    uint64_t terminal_evaluations=0,terminal_leaf_endpoints=0,depth_limit_endpoints=0;
    uint64_t proof_root_updates=0;
    uint64_t attempted_simulations=0,aborted_simulations=0;
    uint64_t prune_count=0,nodes_evicted=0,edges_evicted=0,discarded_neural_evaluations=0;
};
struct TelemetryEvent {
    bool occurred=false;
    std::string reason;
    uint64_t simulation=0; // 1-based simulation; 0 means root initialization
    uint64_t attempt=0,completed_simulations=0;
    double elapsed_ms=0;
    int depth=0;
    size_t nodes_size=0,nodes_capacity=0,edges_size=0,edges_capacity=0,memory_bytes=0;
};
struct TelemetryRootSnapshot {
    Move move{};
    float prior=0,q=0;
    uint64_t visits=0;
};
struct TelemetryRoot:TelemetryRootSnapshot {DepthHistogram leaf_depth{};};
struct TelemetryPrune {
    uint64_t attempt=0,completed_simulations=0,threshold=0;
    double elapsed_ms=0,duration_ms=0;
    int failed_depth=0;
    size_t nodes_before=0,nodes_after=0,edges_before=0,edges_after=0;
    size_t nodes_capacity=0,edges_capacity=0,memory_bytes=0,scratch_bytes=0;
    bool success=false,root_statistics_preserved=false,protected_root_children=false;
    std::vector<TelemetryRootSnapshot> roots_before,roots_after;
};
struct TelemetrySnapshot {
    std::string reason;
    uint64_t simulation=0;
    double elapsed_ms=0;
    DepthSummary depth{},window_depth{};
    DepthHistogram leaf_depth{},window_leaf_depth{};
    size_t nodes_size=0,nodes_capacity=0,edges_size=0,edges_capacity=0,memory_bytes=0;
    TelemetryCounters counters{};
    std::vector<TelemetryRootSnapshot> roots;
};
struct SearchTelemetry {
    DepthHistogram leaf_depth{},traversal_depth{},pre_failure_leaf_depth{},post_failure_leaf_depth{};
    DepthSummary depth{};
    std::vector<TelemetryRoot> roots;
    std::vector<TelemetrySnapshot> snapshots;
    std::vector<TelemetryPrune> prunes;
    TelemetryEvent first_reserved_cap{},first_allocation_failure{};
    TelemetryCounters counters{};
    uint64_t root_initializations=1;
    bool proof_only=false;
    size_t telemetry_memory_bytes=0; // retained telemetry estimate, outside arena budget
};
struct SearchLimits {
    int simulations=800,milliseconds=0;
    float cpuct=1.5f,noise=0,temperature=0,fpu_reduction=0;
    uint64_t seed=1;
    size_t max_nodes=100000,max_tree_bytes=64*1024*1024;
    bool tactical_safety=false;
    int proof_nodes=0,proof_plies=1;
    bool telemetry=false;
    int telemetry_interval=4096;
};
struct SearchResult {
    Move best{};bool has_move=false,terminal=false;
    int simulations=0,neural_evals=0,max_depth=0;
    size_t arena_nodes=0,memory_bytes=0;
    size_t peak_memory_bytes=0,peak_auxiliary_bytes=0;
    uint64_t attempted_simulations=0,aborted_simulations=0,prune_count=0;
    uint64_t nodes_evicted=0,edges_evicted=0,discarded_neural_evaluations=0;
    bool memory_exhausted=false;
    float value=0;
    int proof_nodes=0,proof_plies=0;
    bool mate_proven=false;
    std::vector<std::pair<Move,int>> visits;
    std::vector<float> q_values,priors;
    std::shared_ptr<SearchTelemetry> telemetry;
};
SearchResult search(const Board& board,const Network& network,const SearchLimits& limits,const std::atomic<bool>& stop);
}
