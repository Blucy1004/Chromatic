#pragma once
#include "board.hpp"
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace zero {
inline constexpr int RRN_FEATURES=19;
inline constexpr int RRN_HIDDEN=8;
inline constexpr int RRN_RELATIONS=4;
inline constexpr int RRN_MESSAGE_STEPS=3;

enum class RrnNodeType : uint8_t { EmptySquare, Pawn, Knight, Bishop, Rook, Queen, King };
enum class RrnRelationType : uint8_t { AdjacentSquare=0, SameColor=1, SamePieceType=2, LegalCandidate=3 };
struct RrnNode {
    int square=0;
    RrnNodeType type=RrnNodeType::EmptySquare;
    int color=0;
};
struct RrnEdge {
    uint8_t source=0,target=0;
    RrnRelationType relation=RrnRelationType::AdjacentSquare;
};

// Version 1 text weights are flattened in the documented order in rrn.cpp.
// The shipped reference object is deterministic but untrained; learned=true
// is reserved for externally trained, provenance-documented weights.
struct RrnWeights {
    bool learned=false;
    uint32_t seed=2026;
    std::string provenance="deterministic-reference-v1";
    std::array<float,RRN_FEATURES*RRN_HIDDEN> input{};
    std::array<float,RRN_HIDDEN> input_bias{};
    std::array<float,RRN_HIDDEN*RRN_HIDDEN> self{};
    std::array<float,RRN_RELATIONS*RRN_HIDDEN*RRN_HIDDEN> relation{};
    std::array<float,RRN_HIDDEN> message_bias{};
    std::array<float,2*RRN_HIDDEN+5> rank{};
    float rank_bias=0;

    static RrnWeights deterministic_reference(uint32_t seed=2026);
    static RrnWeights load(const std::string& path);
    void save(const std::string& path) const;
};

class RrnRanker {
public:
    explicit RrnRanker(RrnWeights weights=RrnWeights::deterministic_reference());
    static RrnRanker reference(uint32_t seed=2026) { return RrnRanker(RrnWeights::deterministic_reference(seed)); }
    // Returns advisory scores aligned 1:1 with legal moves. Never prunes.
    std::vector<float> rank(const Board& board,const std::vector<Move>& legal_moves) const;
    std::vector<float> scores(const Board& board,const std::vector<Move>& legal_moves) const { return rank(board,legal_moves); }
    bool learned() const { return weights_.learned; }
    const std::string& provenance() const { return weights_.provenance; }
    static constexpr const char* status() { return "untrained/reference only; ordering hints, no pruning"; }
private:
    RrnWeights weights_;
};

// Small direct-mapped, exact-key cache for repeated PVS ordering calls.
// Cache hits are advisory only; collisions and full-board mismatches recompute.
class RrnOrderCache {
public:
    explicit RrnOrderCache(const RrnRanker& ranker, size_t capacity=4096);
    std::vector<float> rank(const Board& board,const std::vector<Move>& legal_moves);
    uint64_t calls() const { return calls_; }
    uint64_t hits() const { return hits_; }
private:
    struct Entry {
        bool valid=false;
        uint64_t hash=0;
        std::array<int,64> squares{};
        int turn=0,castle=0,ep=-1,halfmove=0;
        std::vector<Move> moves;
        std::vector<float> scores;
    };
    const RrnRanker& ranker_;
    std::vector<Entry> entries_;
    uint64_t calls_=0,hits_=0;
};
}


