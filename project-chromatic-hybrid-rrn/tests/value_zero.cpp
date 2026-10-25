#include "zero/search.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace zero;
namespace {
void require(bool ok,const char* message) {
    if(!ok)throw std::runtime_error(message);
}
bool same_result(const SearchResult& a,const SearchResult& b) {
    return a.best==b.best&&a.has_move==b.has_move&&a.terminal==b.terminal&&
        a.simulations==b.simulations&&a.neural_evals==b.neural_evals&&a.value==b.value&&
        a.visits==b.visits&&a.q_values==b.q_values&&a.priors==b.priors&&
        a.root_neural_value==b.root_neural_value&&a.root_raw_wdl==b.root_raw_wdl&&
        a.root_raw_value==b.root_raw_value&&a.root_search_value==b.root_search_value;
}
SearchResult run(const Board& board,const Network& net,SearchLimits limits) {
    std::atomic<bool> stop=false;
    return search(board,net,limits,stop);
}
void terminal_preserved(const char* fen,float expected) {
    Network net;
    for(bool zero:{false,true}) {
        SearchLimits limits;limits.value_zero=zero;limits.proof_nodes=0;limits.simulations=64;
        auto result=run(Board::from_fen(fen),net,limits);
        require(result.terminal,"terminal board became nonterminal");
        require(!result.root_neural_value&&result.neural_evals==0,"terminal position was sent through the neural value override");
        require(result.value==expected,"terminal mate/draw value was changed");
    }
}
}

int main(int argc,char** argv) {
    try {
        Network net;
        if(argc>1)net.load(argv[1]);
        const Board position=Board::from_fen("rnbqkb1r/pp2pppp/2p2n2/2N5/3P4/2P5/PP2BPPP/R1BQK1NR b KQkq - 2 7");

        // The default field and explicit OFF mode must preserve the old search exactly.
        SearchLimits default_limits;default_limits.simulations=512;default_limits.seed=2026;
        default_limits.proof_nodes=0;default_limits.telemetry=true;
        SearchLimits off_limits=default_limits;off_limits.value_zero=false;
        const auto default_run=run(position,net,default_limits);
        const auto off_run=run(position,net,off_limits);
        require(!default_limits.value_zero,"ValueZero must default OFF");
        require(same_result(default_run,off_run),"default/OFF behavior diverged from legacy search");
        require(off_run.root_neural_value==net.loaded(),"root neural-value flag does not reflect model loading");
        if(!net.loaded())require(off_run.root_raw_wdl==std::array<float,3>{.25f,.5f,.25f},"uniform fallback output changed");
        require(off_run.root_search_value==off_run.root_raw_value,"OFF search value differs from raw WDL scalar");
        require(std::abs((off_run.root_raw_wdl[0]-off_run.root_raw_wdl[2])-off_run.root_raw_value)<1e-6f,"raw scalar is not WDL win-minus-loss");

        // ON changes only the scalar handed to search; WDL and legal policy priors stay untouched.
        SearchLimits on_limits=off_limits;on_limits.value_zero=true;
        const auto on_run=run(position,net,on_limits);
        require(on_run.root_neural_value==net.loaded(),"ValueZero changed model provenance flag");
        require(on_run.root_raw_wdl==off_run.root_raw_wdl,"ValueZero changed raw WDL output");
        require(on_run.root_raw_value==off_run.root_raw_value,"ValueZero changed raw WDL scalar");
        require(on_run.root_search_value==0.0f,"ValueZero did not force nonterminal search scalar to exact zero");
        require(on_run.priors==off_run.priors,"ValueZero changed root policy priors");
        require(on_run.visits.size()==position.legal().size(),"root candidate count differs from legal moves");
        if(net.loaded())require(std::abs(off_run.root_raw_value)>1e-4f,"deployed model fixture did not exercise a nonzero value");

        terminal_preserved("7k/6Q1/6K1/8/8/8/8/8 b - - 100 1",-1.0f);
        terminal_preserved("4k3/8/8/8/8/8/8/4K3 w - - 0 1",0.0f);
        std::cout<<"value-zero regression passed; model="<<(net.loaded()?"deployed":"uniform")
                 <<" raw="<<off_run.root_raw_value<<" forced="<<on_run.root_search_value<<'\n';
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<e.what()<<'\n';return 1;
    }
}
