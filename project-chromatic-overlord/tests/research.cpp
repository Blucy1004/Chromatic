#include "zero/search.hpp"
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
using namespace zero;
void require(bool ok,const char* s){if(!ok)throw std::runtime_error(s);}
int main(int argc,char** argv){try {
 Network net;if(argc>1)net.load(argv[1]);auto b=Board::from_fen(START_FEN);std::atomic<bool> stop=false;
 for(int mask=0;mask<32;++mask){
  SearchLimits l;l.simulations=64;l.counterexample=mask&1;l.adaptive_budget=mask&2;l.forcing_search=mask&4;l.tactical_verify=mask&8;l.threads=(mask&16)?2:1;
  auto plain=search(b,net,l,stop);l.research_metrics=true;auto measured=search(b,net,l,stop);
  require(plain.visits==measured.visits&&plain.q_values==measured.q_values&&plain.best==measured.best,"Research metrics changed fixed-work decisions");
  auto& m=*measured.research;require(std::accumulate(m.leaf_depth.begin(),m.leaf_depth.end(),uint64_t(0))==uint64_t(measured.simulations),"depth endpoints");
  require(m.node_creations+1-measured.nodes_evicted==measured.arena_nodes,"actual node creation accounting");
  require(m.edge_expansions>0&&m.nn_mcts_ns>0&&m.algorithm_ns>0,"phase metrics execute");
  require(!m.pv.empty()&&m.pv.front()==measured.best,"selected PV starts with final choice");
  l.neural_limit=32;l.simulations=32*16;auto capped=search(b,net,l,stop);
  require(capped.neural_evals+capped.verification_evals<=32,"total NN cap includes verifier");
 }
 for(int cap:{1,2,16}){SearchLimits l;l.neural_limit=cap;l.simulations=cap*16;l.counterexample=l.forcing_search=l.tactical_verify=true;l.research_metrics=true;auto r=search(b,net,l,stop);require(r.neural_evals+r.verification_evals<=cap&&r.has_move,"small NN budget stops safely");}
 std::cout<<"32-cell metrics parity, depth/counts/PV and total NN cap passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
