#pragma once
#include "search.hpp"
namespace zero {
// SANABI and Laugh Tale are new designs in this revision, not recovered code.
float budget_importance(const std::vector<float>& priors,float value);
struct VerificationResult {bool complete=false;float value=0;int nodes=0,neural_evals=0;};
VerificationResult verify_candidate(const Board&,Move,const Network&,int depth,int nodes,
    bool counterexample,bool forcing,bool tactical,const std::atomic<bool>&,
    std::chrono::steady_clock::time_point deadline);
void verify_candidates(const Board&,const Network&,const SearchLimits&,const std::atomic<bool>&,
    std::chrono::steady_clock::time_point deadline,SearchResult&);
}
