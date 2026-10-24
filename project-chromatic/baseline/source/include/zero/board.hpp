#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace zero {
inline constexpr const char* START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
inline constexpr int INPUTS = 20 * 64;
inline constexpr int ACTIONS = 4096 + 64 * 9;
struct Move {
    int from=0, to=0, promo=0; // 2=N,3=B,4=R,5=Q
    std::string uci() const;
    bool operator==(const Move&) const = default;
};
int action(Move m);
struct Board {
    std::array<int,64> squares{}; // white +, black -; pawn=1 ... king=6
    int turn=1, castle=0, ep=-1, halfmove=0, fullmove=1;
    std::vector<uint64_t> history;
    static Board from_fen(const std::string& fen);
    std::string fen() const;
    bool attacked(int square, int by) const;
    bool check(int color) const;
    std::vector<Move> legal() const;
    void make(Move move, bool track=true);
    void play(const std::string& uci);
    uint64_t key() const;
    int repetitions() const;
    bool insufficient() const;
    // Caller supplies legal moves, so mate takes precedence over move-count draws.
    bool terminal(const std::vector<Move>& moves, float& value) const;
    std::array<float,INPUTS> features() const;
};
uint64_t perft(const Board& board, int depth);
}
