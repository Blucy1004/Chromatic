#include "zero/search.hpp"
#include "zero/profile.hpp"
#include <iomanip>
#include <iostream>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#endif
using namespace zero;
int main(int argc,char** argv){try {
#ifdef _WIN32
    if(!SetProcessAffinityMask(GetCurrentProcess(),DWORD_PTR(1)))throw std::runtime_error("cannot pin benchmark to CPU 0");
#endif
    if(argc<2)throw std::runtime_error("native_bench MODEL [EVALS=1000] [SIMULATIONS=128]");
    int evals=argc>2?std::stoi(argv[2]):1000,sims=argc>3?std::stoi(argv[3]):128;
    if(evals<1||evals>100000||sims<1||sims>100000)throw std::runtime_error("invalid benchmark budget");
    Network net;net.load(argv[1]);
#ifndef CHROMATIC_BASELINE
    if(argc>4)net.set_backend(std::string(argv[4])=="scalar"?InferenceBackend::Scalar:InferenceBackend::SIMD);
    if(argc>5)net.set_sparse(std::string(argv[5])!="full");
#endif
    const char* fens[]={START_FEN,
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
      "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
      "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"};
    std::vector<Board> boards;for(auto f:fens)boards.push_back(Board::from_fen(f));
    double checksum=0;for(int i=0;i<16;++i)checksum+=net.evaluate(boards[i%4]).wdl[0];
    auto start=std::chrono::steady_clock::now();
    for(int i=0;i<evals;++i){auto e=net.evaluate(boards[i%4]);checksum+=e.policy[i%ACTIONS]+e.wdl[0];}
    double inference=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
#ifndef CHROMATIC_BASELINE
    std::vector<std::vector<Move>> legal;for(auto& board:boards)legal.push_back(board.legal());
    Network::Workspace workspace;start=std::chrono::steady_clock::now();
    for(int i=0;i<evals;++i){auto e=net.evaluate_legal(boards[i%4],legal[i%4],workspace);checksum+=e.wdl[0];}
    double legal_inference=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
#endif
    std::atomic<bool> stop=false;SearchLimits limits;limits.simulations=sims;limits.seed=2026;
    profile::reset();profile::active=true;start=std::chrono::steady_clock::now();
    int nodes=0;for(auto& b:boards){auto r=search(b,net,limits,stop);nodes+=r.simulations;checksum+=r.value;}
    double search_time=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();profile::active=false;
    std::cout<<std::setprecision(12)<<"{\"affinity_mask\":1,\"evaluations\":"<<evals<<",\"simulations\":"<<nodes<<",\"positions\":4,\"inference_seconds\":"<<inference
      <<",\"mean_inference_us\":"<<inference*1e6/evals<<",\"evaluations_per_second\":"<<evals/inference
      <<",\"search_seconds\":"<<search_time<<",\"nodes_per_second\":"<<nodes/search_time
      <<",\"search_neural_calls\":"<<profile::neural_calls<<",\"search_neural_seconds\":"<<profile::neural_ns*1e-9
      <<",\"search_overhead_fraction\":"<<(search_time-profile::neural_ns*1e-9)/search_time
      <<",\"movegen_seconds\":"<<profile::movegen_ns*1e-9<<",\"feature_seconds\":"<<profile::feature_ns*1e-9<<",\"peak_working_set_bytes\":";
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc{};if(GetProcessMemoryInfo(GetCurrentProcess(),&pmc,sizeof(pmc)))std::cout<<pmc.PeakWorkingSetSize;else std::cout<<"null";
#else
    std::cout<<"null";
#endif
#ifndef CHROMATIC_BASELINE
    std::cout<<",\"legal_mean_inference_us\":"<<legal_inference*1e6/evals<<",\"legal_evaluations_per_second\":"<<evals/legal_inference;
#endif
    std::cout<<",\"checksum\":"<<checksum<<"}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
