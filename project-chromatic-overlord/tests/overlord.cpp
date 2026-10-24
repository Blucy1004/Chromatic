#include "zero/verification.hpp"
#include <cmath>
#include <future>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>
using namespace zero;
void require(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
int main(int argc,char** argv){try {
    Network net;if(argc>1)net.load(argv[1]);std::atomic<bool> stop=false;
    auto start=Board::from_fen(START_FEN);SearchLimits l;l.simulations=256;
    {
        std::ofstream f("parallel-sign.zero",std::ios::binary);f.write("ZERO0001",8);
        uint32_t header[]={INPUTS,1,0,ACTIONS};f.write(reinterpret_cast<const char*>(header),sizeof(header));
        auto zeros=[&](int n){float z=0;for(int i=0;i<n;++i)f.write(reinterpret_cast<const char*>(&z),4);};
        zeros(INPUTS+1);zeros(ACTIONS+ACTIONS);zeros(3);float biases[]={8,-8,-8};f.write(reinterpret_cast<const char*>(biases),sizeof(biases));f.close();
        Network child_wins;child_wins.load("parallel-sign.zero");
        for(int threads:{1,2,4}){auto one=l;one.threads=threads;one.simulations=threads;
            auto r=search(start,child_wins,one,stop);require(r.value<-.99f,"parallel WDL parent-child inversion");}
        std::filesystem::remove("parallel-sign.zero");
    }
    auto legacy=search(start,net,l,stop);auto again=search(start,net,l,stop);
    require(legacy.visits==again.visits&&legacy.q_values==again.q_values,"disabled path reproducible");
    require(budget_importance({.5f,.5f},0)>budget_importance({.999f,.001f},.9f),"importance responds to entropy and uncertainty");
    for(int threads:{2,4})for(bool measured:{false,true}) {
        l.threads=threads;l.telemetry=measured;l.max_nodes=16;l.max_tree_bytes=65536;l.simulations=512;
        auto r=search(start,net,l,stop);
        require(r.simulations==512,"parallel exact simulation budget");
        int n=0;for(auto [m,v]:r.visits)n+=v;for(float q:r.q_values)require(std::isfinite(q)&&std::abs(q)<=1,"finite bounded Q");
        require(n==r.simulations,"virtual losses leave no fabricated root visits");
        require(r.prune_count>0&&r.memory_bytes<=l.max_tree_bytes,"parallel memory reclamation");
        require(r.attempted_simulations==uint64_t(r.simulations)+r.aborted_simulations,"parallel aborted visits");
        require(r.peak_inflight>=2,"actual concurrent neural leaves");
        if(measured) {
            require(r.telemetry->depth.samples==uint64_t(r.simulations),"parallel endpoint accounting");
            require(r.telemetry->counters.resident_unexpanded_repeat==0,"no repeated resident NN evaluation");
            for(auto& prune:r.telemetry->prunes)require(prune.root_statistics_preserved,"parallel prune preserves root stats");
        }
    }
    l=SearchLimits{};l.simulations=256;
    auto mate=Board::from_fen("7k/8/5KQ1/8/8/8/8/8 w - - 0 1");
    for(int threads:{1,2,4}) {
        l.threads=threads;Network uniform;auto r=search(mate,uniform,l,stop);auto after=mate;after.make(r.best);float v=0;
        require(after.terminal(after.legal(),v)&&v==-1,"parallel PUCT mate sign");
    }
    l.proof_nodes=256;
    auto proven=search(mate,net,l,stop);require(proven.mate_proven&&proven.value==1,"loaded model exact mate proof");
    auto deadline=std::chrono::steady_clock::time_point::max();
    auto hanging=Board::from_fen("6k1/8/8/8/8/8/3q4/3Q2K1 w - - 0 1");
    for(int mode=0;mode<3;++mode) {
        auto good=verify_candidate(hanging,Move{3,11,0},net,4,4096,mode==0,mode==1,mode==2,stop,deadline);
        auto bad=verify_candidate(hanging,Move{3,4,0},net,4,4096,mode==0,mode==1,mode==2,stop,deadline);
        require(good.complete&&bad.complete,"bounded verifier completes comparison");
        if(mode==2)require(good.value>bad.value+.2,"tactical verifier catches hanging queen");
    }
    auto incomplete=verify_candidate(start,start.legal()[0],net,4,1,true,true,true,stop,deadline);
    require(!incomplete.complete&&incomplete.nodes<=1,"incomplete verification cannot certify a candidate");
    l=SearchLimits{};l.simulations=128;l.counterexample=true;l.verification_nodes=1;
    auto partial=search(start,net,l,stop);l.counterexample=false;
    auto base=search(start,net,l,stop);require(base.best==partial.best&&!partial.verification_changed,"partial comparisons preserve decision");
    l=SearchLimits{};l.threads=4;l.simulations=0;l.milliseconds=30;l.counterexample=true;l.forcing_search=true;l.tactical_verify=true;
    auto t=std::chrono::steady_clock::now();auto timed=search(start,net,l,stop);
    require(timed.has_move&&std::chrono::steady_clock::now()-t<std::chrono::seconds(2),"bounded clock with verification");
    l.milliseconds=0;
    auto job=std::async(std::launch::async,[&]{return search(start,net,l,stop);});
    std::this_thread::sleep_for(std::chrono::milliseconds(10));stop=true;
    require(job.wait_for(std::chrono::seconds(2))==std::future_status::ready,"stop joins parallel workers");job.get();stop=false;
    for(auto fen:{"7k/6Q1/6K1/8/8/8/8/8 b - - 100 1","7k/5Q2/6K1/8/8/8/8/8 b - - 0 1","4k3/8/8/8/8/8/P7/4K3 w - - 100 1"}) {
        auto r=search(Board::from_fen(fen),net,l,stop);require(r.terminal&&!r.has_move&&r.simulations==0,"terminal roots precede verification");
    }
    auto repetition=start;for(int i=0;i<2;++i)for(auto m:{"g1f3","g8f6","f3g1","f6g8"})repetition.play(m);
    require(search(repetition,net,l,stop).terminal,"parallel repetition history");
    std::cout<<"Overlord module, sign, memory, terminal and cancellation tests passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
