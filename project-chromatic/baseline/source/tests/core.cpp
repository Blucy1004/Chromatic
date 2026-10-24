#include "zero/search.hpp"
#include <iostream>
#include <stdexcept>
#include <set>
using namespace zero;
void require(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
int main(){try {
    struct Case {const char* fen;int depth;uint64_t count;};
    Case cases[]={
      {START_FEN,5,4865609},
      {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",4,4085603},
      {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",5,674624},
      {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",4,422333},
      {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",4,2103487}
    };
    for(auto c:cases){auto b=Board::from_fen(c.fen);auto n=perft(b,c.depth);std::cout<<"perft "<<c.depth<<" "<<n<<std::endl;require(n==c.count,"perft mismatch");require(b.fen()==c.fen,"FEN roundtrip");}
    Board b=Board::from_fen(START_FEN);for(int i=0;i<2;++i)for(auto s:{"g1f3","g8f6","f3g1","f6g8"})b.play(s);require(b.repetitions()==3,"threefold");float value=4;require(b.terminal(b.legal(),value)&&value==0,"repetition draw");
    auto a=Board::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1");require(a.insufficient(),"dead position");
    a=Board::from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");require(a.terminal(a.legal(),value)&&value==0,"stalemate");
    a=Board::from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 100 1");require(a.terminal(a.legal(),value)&&value==-1,"mate precedes 50 moves");
    a=Board::from_fen("4k3/8/8/r4pPK/8/8/8/8 w - f6 0 1");auto noep=Board::from_fen("4k3/8/8/r4pPK/8/8/8/8 w - - 0 1");require(a.key()==noep.key(),"pinned EP repetition key");
    for(auto m:a.legal())require(m.uci()!="g5f6","EP discovered check");
    a=Board::from_fen("1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1");std::set<int> ids;for(auto m:a.legal()){require(action(m)>=0&&action(m)<ACTIONS,"action range");require(ids.insert(action(m)).second,"action collision");}
    Network net;std::atomic<bool> stop=false;SearchLimits limits;limits.simulations=128;limits.max_nodes=16;
    a=Board::from_fen("7k/8/5KQ1/8/8/8/8/8 w - - 0 1");auto r=search(a,net,limits,stop);require(r.simulations==128,"simulation budget");int visits=0;for(auto [m,n]:r.visits)visits+=n;require(visits==128,"visit accounting");a.make(r.best);require(a.terminal(a.legal(),value)&&value==-1,"PUCT mate in one / backup sign");
    std::cout<<"all core tests passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
