#include "zero/search.hpp"
#include <iostream>
#include <stdexcept>
#include <set>
#include <fstream>
#include <cmath>
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
    // A child that predicts a win for its side must be a loss for the root side.
    // This assertion fails if the single minus sign in backup is removed.
    {
      std::ofstream f("synthetic-wdl.zero",std::ios::binary);f.write("ZERO0001",8);
      uint32_t header[]={INPUTS,1,0,ACTIONS};f.write(reinterpret_cast<char*>(header),sizeof(header));
      auto zeros=[&](int n){float z=0;for(int i=0;i<n;++i)f.write(reinterpret_cast<char*>(&z),4);};
      zeros(INPUTS+1);zeros(ACTIONS+ACTIONS);zeros(3);float biases[]={8,-8,-8};f.write(reinterpret_cast<char*>(biases),sizeof(biases));f.close();
      Network win_for_child;win_for_child.load("synthetic-wdl.zero");SearchLimits one;one.simulations=1;
      auto sign=search(Board::from_fen(START_FEN),win_for_child,one,stop);require(sign.value<-.99f,"WDL child-to-parent sign inversion");
    }
    for(auto fen:{"8/8/8/8/8/8/4K3/7k w K - 0 1","4k3/8/8/8/8/8/8/4K3 w - - 2147483647 1"}){
      bool rejected=false;try{Board::from_fen(fen);}catch(const std::exception&){rejected=true;}require(rejected,"invalid FEN must be rejected");
    }
    a=Board::from_fen("7k/8/5KQ1/8/8/8/8/8 w - - 0 1");auto r=search(a,net,limits,stop);require(r.simulations==128,"simulation budget");int visits=0;for(auto [m,n]:r.visits)visits+=n;require(visits==128,"visit accounting");a.make(r.best);require(a.terminal(a.legal(),value)&&value==-1,"PUCT mate in one / backup sign");
    limits.max_nodes=100000;limits.max_tree_bytes=65536;limits.simulations=512;
    auto bounded=search(Board::from_fen(START_FEN),net,limits,stop);
    require(bounded.simulations==512&&bounded.memory_bytes<=65536,"tree byte budget and progress");
    require(bounded.prune_count>0&&bounded.nodes_evicted>0,"byte-capacity pressure must reclaim stored tree slots");
    require(bounded.attempted_simulations==uint64_t(bounded.simulations)+bounded.aborted_simulations,"failed expansions must not invent visits");
    require(bounded.peak_memory_bytes<=65536&&!bounded.memory_exhausted,"recycling stays inside the reserved tree byte cap");
    limits.noise=.25f;limits.temperature=1;limits.seed=81;
    auto deterministic=search(Board::from_fen(START_FEN),net,limits,stop);
    auto repeat_search=search(Board::from_fen(START_FEN),net,limits,stop);
    require(deterministic.visits==repeat_search.visits&&deterministic.best==repeat_search.best,"seeded search reproducibility");
    float priors=0;for(float p:deterministic.priors){require(std::isfinite(p)&&p>=0,"valid prior");priors+=p;}require(std::abs(priors-1)<1e-5,"normalized legal prior");
    limits.noise=0;limits.temperature=0;limits.simulations=1;limits.proof_nodes=12000;limits.proof_plies=5;
    auto proof=search(Board::from_fen("7k/8/8/4K3/8/2Q5/8/8 w - - 0 1"),net,limits,stop);
    require(proof.mate_proven&&proof.proof_plies==5&&proof.proof_nodes<=12000,"bounded mate in three proof");
    require(proof.best.uci()=="c3g3"||proof.best.uci()=="c3c7"||proof.best.uci()=="e5f6","independently verified mate solution");
    require(proof.simulations==1,"proof must not invent unused simulations");
    limits.proof_nodes=1;
    auto unknown=search(Board::from_fen(START_FEN),net,limits,stop);
    require(!unknown.mate_proven&&unknown.proof_nodes<=1,"exhausted proof is unknown, not a win");
    std::cout<<"all core tests passed\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
