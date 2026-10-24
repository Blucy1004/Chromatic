#pragma once
#include "board.hpp"
#include <atomic>
#include <chrono>
namespace zero {
struct MateProof {bool proven=false;Move move{};int plies=0,nodes=0;};
// Bounded terminal-only AND/OR proof. No material score, quiescence or alpha-beta.
MateProof verify_mate(const Board& board,const std::vector<Move>& moves,int plies,int nodes,
                      const std::atomic<bool>& stop,int microseconds);
}
