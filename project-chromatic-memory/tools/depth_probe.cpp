#ifdef CHROMATIC_PRISTINE_PROBE
#include "zero/search.hpp"
#else
#include "zero/telemetry_json.hpp"
#endif
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif
using namespace zero;
int main(int argc,char** argv){try {
    std::map<std::string,std::string> args;
    for(int i=1;i<argc;i+=2){if(i+1>=argc)throw std::runtime_error("expected --name VALUE pairs");args[argv[i]]=argv[i+1];}
    auto get=[&](const std::string& k,const std::string& fallback){auto it=args.find(k);return it==args.end()?fallback:it->second;};
    const auto model=get("--model","");if(model.empty())throw std::runtime_error("--model FILE required");
    auto threads=std::stoi(get("--threads","1"));if(threads!=1)throw std::runtime_error("only one search thread is implemented");
    auto affinity=std::stoull(get("--affinity","2"));
#ifdef _WIN32
    if(!SetProcessAffinityMask(GetCurrentProcess(),DWORD_PTR(affinity)))throw std::runtime_error("cannot set process affinity");
#endif
    SearchLimits limits;limits.seed=std::stoull(get("--seed","2026"));
    auto memory=std::stoull(get("--memory","64"));if(memory<1||memory>2048)throw std::runtime_error("memory MiB out of range");
    limits.max_tree_bytes=memory*1024*1024;limits.max_nodes=std::stoull(get("--max-nodes","1000000"));
    if(limits.max_nodes<1||limits.max_nodes>1000000)throw std::runtime_error("max nodes out of range");
#ifndef CHROMATIC_PRISTINE_PROBE
    limits.telemetry=get("--telemetry","true")=="true";
    limits.telemetry_interval=std::stoi(get("--interval","4096"));
    if(limits.telemetry_interval<64||limits.telemetry_interval>1000000)throw std::runtime_error("interval out of range");
#endif
    const auto mode=get("--mode","simulations");auto budget=std::stoi(get("--budget","4096"));if(budget<1||budget>10000000)throw std::runtime_error("budget out of range");
    if(mode=="time"){limits.simulations=0;limits.milliseconds=budget;}else if(mode=="simulations"){limits.simulations=budget;limits.milliseconds=0;}else throw std::runtime_error("mode must be time or simulations");
    limits.noise=0;limits.temperature=0;limits.proof_nodes=0;
    Network network;network.load(model);network.set_backend(InferenceBackend::SIMD);network.set_sparse(true);
    Board board=Board::from_fen(get("--fen",START_FEN));std::atomic<bool> stop=false;
    if(get("--warmup","true")=="true"){
        Network::Workspace workspace;auto legal=board.legal();for(int i=0;i<32;++i)(void)network.evaluate_legal(board,legal,workspace);
    }
    auto start=std::chrono::steady_clock::now();auto result=search(board,network,limits,stop);
    auto elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
#ifdef CHROMATIC_PRISTINE_PROBE
    std::string payload="{\"bestmove\":\""+result.best.uci()+"\",\"simulations\":"+std::to_string(result.simulations)+",\"max_depth\":"+std::to_string(result.max_depth)+"}";
#else
    auto payload=search_result_json(result);
#endif
    std::cout<<std::setprecision(12)<<"{\"elapsed_ms\":"<<elapsed<<",\"affinity_mask\":"<<affinity<<",\"threads\":1,\"result\":"<<payload<<"}\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
