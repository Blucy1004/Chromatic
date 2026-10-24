#include "zero/proof.hpp"
#include <algorithm>
namespace zero {
namespace {
enum class Proof {Refuted,Unknown,Win};
struct Context {
    int attacker,budget,nodes=0,microseconds;
    const std::atomic<bool>& stop;
    std::chrono::steady_clock::time_point start=std::chrono::steady_clock::now();
    bool expired() const {
        return nodes>=budget||stop.load(std::memory_order_relaxed)||
          (microseconds>0&&std::chrono::steady_clock::now()-start>=std::chrono::microseconds(microseconds));
    }
    Proof forced(const Board& b,int remaining) {
        if(expired())return Proof::Unknown;
        ++nodes;
        if(remaining==0&&!b.check(b.turn))return Proof::Refuted;
        auto moves=b.legal();float value;
        if(b.terminal(moves,value))return value==-1&&b.turn!=attacker?Proof::Win:Proof::Refuted;
        if(remaining==0)return Proof::Refuted;
        std::stable_partition(moves.begin(),moves.end(),[&](Move move){auto c=b.position_only();c.make(move,false);return c.check(c.turn);});
        bool unknown=false;
        for(auto move:moves) {
            if(expired())return Proof::Unknown;
            Board child=b;child.make(move);Proof proof=forced(child,remaining-1);
            if(b.turn==attacker&&proof==Proof::Win)return Proof::Win;
            if(b.turn!=attacker&&proof==Proof::Refuted)return Proof::Refuted;
            unknown|=proof==Proof::Unknown;
        }
        if(unknown)return Proof::Unknown;
        return b.turn==attacker?Proof::Refuted:Proof::Win;
    }
};
}
MateProof verify_mate(const Board& b,const std::vector<Move>& moves,int plies,int nodes,const std::atomic<bool>& stop,int microseconds) {
    Context context{b.turn,nodes,0,microseconds,stop};MateProof result;
    for(int depth=1;depth<=plies;depth+=2) {
        for(auto move:moves) {
            if(context.expired()){result.nodes=context.nodes;return result;}
            Board child=b;child.make(move);
            if(context.forced(child,depth-1)==Proof::Win){result={true,move,depth,context.nodes};return result;}
        }
    }
    result.nodes=context.nodes;return result;
}
}
