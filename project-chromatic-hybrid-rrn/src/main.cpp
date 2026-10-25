#include "zero/search.hpp"
#include "zero/pvs.hpp"
#include "zero/rrn.hpp"
#include "zero/telemetry_json.hpp"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <thread>

using namespace zero;
namespace {
std::mutex output_mutex;
void say(const std::string& s){std::lock_guard lock(output_mutex);std::cout<<s<<std::endl;}
std::string escape(const std::string& s){std::string r;for(unsigned char c:s){if(c=='"'||c=='\\')r+='\\';if(c>=32)r+=char(c);}return r;}
template<class T>void array(std::ostream& o,const T& values){o<<'[';bool first=true;for(auto v:values){if(!first)o<<',';first=false;o<<v;}o<<']';}
std::string inspect(const Board& b,const Network& net,bool evaluation,bool legal_only=false){
    std::ostringstream o;o<<std::setprecision(9);auto moves=b.legal();float value=0;bool terminal=b.terminal(moves,value);
    o<<"{\"fen\":\""<<b.fen()<<"\",\"repetitions\":"<<b.repetitions()<<",\"terminal\":"<<(terminal?"true":"false")<<",\"value\":"<<value<<",\"features\":";array(o,b.features());
    o<<",\"legal\":[";bool first=true;for(auto m:moves){if(!first)o<<',';first=false;o<<"{\"uci\":\""<<m.uci()<<"\",\"action\":"<<action(m)<<'}';}o<<']';
    if(evaluation){Network::Workspace workspace;auto e=legal_only?net.evaluate_legal(b,moves,workspace):net.evaluate(b);o<<",\"policy\":";array(o,e.policy);o<<",\"wdl\":";array(o,e.wdl);}
    o<<'}';return o.str();
}
std::string result_json(const SearchResult& r){return search_result_json(r);}

Board position(std::istringstream& in){
    std::string word;in>>word;Board b;
    if(word=="startpos")b=Board::from_fen(START_FEN);
    else if(word=="fen"){std::string fen,field;for(int i=0;i<6;++i){if(!(in>>field))throw std::runtime_error("incomplete FEN");if(i)fen+=' ';fen+=field;}b=Board::from_fen(fen);}
    else throw std::runtime_error("expected startpos or fen");
    if(in>>word){if(word!="moves")throw std::runtime_error("expected moves");while(in>>word)b.play(word);}return b;
}
}
int main(int argc,char** argv){
    try {
        std::string model_path;
        auto adjacent=std::filesystem::absolute(argv[0]).parent_path()/"default.zero";
        if(std::filesystem::exists(adjacent))model_path=adjacent.string();
        if(argc>1){
            if(std::string(argv[1])=="--perft"&&argc>=3){int d=std::stoi(argv[2]);if(d<0||d>8)throw std::runtime_error("perft depth must be 0..8");std::cout<<perft(Board::from_fen(argc>=4?argv[3]:START_FEN),d)<<std::endl;return 0;}
            if(std::string(argv[1])=="--model"&&argc==3)model_path=argv[2];
            else throw std::runtime_error("usage: project_zero [--model FILE] | --perft DEPTH [FEN]");
        }
        Board board=Board::from_fen(START_FEN);Network network;std::string startup_error;
        if(!model_path.empty()){try{network.load(model_path);}catch(const std::exception& e){startup_error=e.what();model_path.clear();}}
        SearchLimits defaults;defaults.proof_nodes=256;std::string search_mode="PVS";bool use_rrn=false;int pvs_depth=12,pvs_hash_mb=32,rrn_min_depth=4;uint64_t pvs_node_limit=1000000;std::optional<RrnWeights> rrn_weights;std::atomic<bool> stop{false};std::thread worker;
        auto halt=[&](){stop.store(true);if(worker.joinable())worker.join();};
        std::string line;
        while(std::getline(std::cin,line)) {
            std::istringstream in(line);std::string cmd;in>>cmd;
            try {
                if(cmd=="quit"){halt();break;}
                if(cmd=="uci") {say("id name PROJECT CHROMATIC Hybrid (experimental PVS + retained MCTS)\nid author PROJECT CHROMATIC\noption name SearchMode type combo default PVS var PVS var MCTS\noption name UseRRN type check default false\noption name RRNWeights type string default <none>\noption name RRNMinDepth type spin default 4 min 1 max 12\noption name PVSMaxDepth type spin default 12 min 1 max 64\noption name PVSNodeLimit type spin default 1000000 min 1 max 1000000000\noption name PVSHashMB type spin default 32 min 1 max 256\noption name ModelFile type string default "+(model_path.empty()?"<none>":model_path)+"\noption name CPuct type string default 1.5\noption name FPUReduction type string default 0\noption name ValueZero type check default false\noption name InferenceBackend type combo default SIMD var Scalar var SIMD\noption name SparsePolicyHead type check default true\noption name MateProofNodes type spin default 256 min 0 max 50000\noption name MateProofPlies type combo default 1 var 1 var 3 var 5\noption name TreeMemoryMB type spin default 64 min 1 max 2048\noption name Simulations type spin default 800 min 1 max 10000000\noption name MaxNodes type spin default 100000 min 16 max 1000000\noption name SearchTelemetry type check default false\noption name TelemetryInterval type spin default 4096 min 64 max 1000000\nuciok");}
                else if(cmd=="isready"){
                    if(!startup_error.empty()){say("info string error "+escape(startup_error));startup_error.clear();}
                    if(!network.loaded())say("info string warning: no model loaded; using uniform WDL/policy fallback");
                    if(use_rrn){const auto w=rrn_weights.value_or(RrnWeights::deterministic_reference());say(std::string("info string RRN ordering enabled; learned=")+(w.learned?"true":"false")+"; "+w.provenance);}
                    say("readyok");
                }
                else if(cmd=="stop")halt();
                else if(cmd=="ucinewgame"){halt();board=Board::from_fen(START_FEN);}
                else if(cmd=="position"){halt();board=position(in);}
                else if(cmd=="setoption") {
                    halt();std::string token,name,value;in>>token;if(token!="name")throw std::runtime_error("missing option name");
                    while(in>>token&&token!="value"){if(!name.empty())name+=' ';name+=token;}
                    std::getline(in,value);if(!value.empty()&&value.front()==' ')value.erase(0,1);
                    if(name=="SearchMode"){if(value!="PVS"&&value!="MCTS")throw std::runtime_error("SearchMode must be PVS or MCTS");search_mode=value;}
                    else if(name=="UseRRN"){if(value!="true"&&value!="false")throw std::runtime_error("UseRRN requires true or false");use_rrn=value=="true";}
                    else if(name=="RRNWeights"){if(value.empty()||value=="<none>")rrn_weights.reset();else rrn_weights=RrnWeights::load(value);say(std::string("info string RRN weights ")+(rrn_weights?(rrn_weights->learned?"learned":"untrained/reference only"):"none"));}
                    else if(name=="RRNMinDepth")rrn_min_depth=std::clamp(std::stoi(value),1,12);
                    else if(name=="PVSMaxDepth")pvs_depth=std::clamp(std::stoi(value),1,64);
                    else if(name=="PVSNodeLimit")pvs_node_limit=uint64_t(std::clamp(std::stoll(value),1LL,1000000000LL));
                    else if(name=="PVSHashMB")pvs_hash_mb=std::clamp(std::stoi(value),1,256);
                    else if(name=="ModelFile"){if(value=="<none>"||value.empty()){auto backend=network.backend();auto sparse=network.sparse();network=Network{};network.set_backend(backend);network.set_sparse(sparse);model_path.clear();}else {network.load(value);model_path=value;}say("info string model "+std::string(network.loaded()?"loaded":"uniform baseline"));}
                    else if(name=="CPuct"){float v=std::stof(value);if(!std::isfinite(v)||v<=0||v>10)throw std::runtime_error("CPuct must be in (0,10]");defaults.cpuct=v;}
                    else if(name=="Simulations")defaults.simulations=std::clamp(std::stoi(value),1,10000000);
                    else if(name=="MaxNodes")defaults.max_nodes=size_t(std::clamp(std::stoi(value),16,1000000));
                    else if(name=="SearchTelemetry"){if(value!="true"&&value!="false")throw std::runtime_error("SearchTelemetry requires true or false");defaults.telemetry=value=="true";}
                    else if(name=="TelemetryInterval")defaults.telemetry_interval=std::clamp(std::stoi(value),64,1000000);
                    else if(name=="TreeMemoryMB")defaults.max_tree_bytes=size_t(std::clamp(std::stoi(value),1,2048))*1024*1024;
                    else if(name=="FPUReduction"){float v=std::stof(value);if(!std::isfinite(v)||v<0||v>1)throw std::runtime_error("FPU reduction must be in [0,1]");defaults.fpu_reduction=v;}
                    else if(name=="ValueZero"){if(value!="true"&&value!="false")throw std::runtime_error("ValueZero requires true or false");defaults.value_zero=value=="true";}
                    else if(name=="InferenceBackend"){if(value!="Scalar"&&value!="SIMD")throw std::runtime_error("unknown inference backend");network.set_backend(value=="Scalar"?InferenceBackend::Scalar:InferenceBackend::SIMD);}
                    else if(name=="SparsePolicyHead"){if(value!="true"&&value!="false")throw std::runtime_error("SparsePolicyHead requires true or false");network.set_sparse(value=="true");}
                    else if(name=="MateProofNodes")defaults.proof_nodes=std::clamp(std::stoi(value),0,50000);
                    else if(name=="MateProofPlies"){int n=std::stoi(value);if(n<1||n>5||n%2==0)throw std::runtime_error("mate proof plies must be 1,3,5");defaults.proof_plies=n;}
                } else if(cmd=="go") {
                    halt();SearchLimits limits=defaults;std::string token;int remaining=0,increment=0,moves_to_go=30,requested_depth=0;uint64_t requested_nodes=0;bool explicit_nodes=false,explicit_time=false,infinite=false,has_clock=false;
                    while(in>>token){if(token=="infinite"){infinite=true;continue;}if(token=="ponder"||token=="searchmoves")throw std::runtime_error("unsupported go option: "+token);long long n;if(!(in>>n))throw std::runtime_error("missing go limit");n=std::clamp(n,0LL,1000000000LL);
                        if(token=="nodes"){limits.simulations=int(std::max(1LL,n));requested_nodes=uint64_t(std::max(1LL,n));explicit_nodes=true;}
                        else if(token=="movetime"){limits.milliseconds=int(std::max(1LL,n));explicit_time=true;}
                        else if(token==(board.turn==1?"wtime":"btime")){remaining=int(n);has_clock=true;}
                        else if(token==(board.turn==1?"winc":"binc"))increment=int(n);
                        else if(token=="movestogo")moves_to_go=std::max(1,int(n));
                        else if(token=="depth") {requested_depth=int(std::clamp(n,1LL,64LL));limits.simulations=int(std::clamp(n*100,1LL,10000000LL));explicit_nodes=true;}
                    }
                    if(!explicit_time&&has_clock)limits.milliseconds=std::max(1,std::min(remaining/2,remaining/moves_to_go+increment/2));
                    if(infinite){limits.simulations=0;limits.milliseconds=0;}else if(limits.milliseconds>0&&!explicit_nodes)limits.simulations=0;
                    stop.store(false);auto copy=board;auto mode=search_mode;auto weights=rrn_weights;auto rrn_enabled=use_rrn;auto depth_default=pvs_depth;auto pvs_nodes_default=pvs_node_limit;auto hash_default=pvs_hash_mb;auto rrn_depth_default=rrn_min_depth;
                    worker=std::thread([&,copy,limits,mode,weights,rrn_enabled,requested_depth,requested_nodes,depth_default,pvs_nodes_default,hash_default,rrn_depth_default,infinite,explicit_nodes,explicit_time](){try {auto start=std::chrono::steady_clock::now();if(mode=="MCTS"){auto r=search(copy,network,limits,stop);auto ms=std::max<int64_t>(1,std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start).count());say("info nodes "+std::to_string(r.simulations)+" time "+std::to_string(ms)+" nps "+std::to_string(r.simulations*1000LL/ms)+" score cp "+std::to_string(int(400*r.value))+" pv "+(r.has_move?r.best.uci():"0000"));if(r.telemetry){std::ostringstream data;write_telemetry_json(data,*r.telemetry);say("info string telemetry "+data.str());}say("bestmove "+(r.has_move?r.best.uci():"0000"));}else{PvsLimits pl;pl.max_depth=requested_depth?requested_depth:(explicit_nodes?64:depth_default);pl.max_nodes=requested_nodes?requested_nodes:pvs_nodes_default;pl.milliseconds=limits.milliseconds;if(infinite){pl.max_depth=64;pl.max_nodes=1000000000ULL;pl.milliseconds=0;}pl.tt_megabytes=size_t(hash_default);pl.ordering_hint_min_depth=rrn_depth_default;std::optional<RrnRanker> ranker;std::unique_ptr<RrnOrderCache> rrn_cache;if(rrn_enabled){ranker.emplace(weights.value_or(RrnWeights::deterministic_reference()));rrn_cache=std::make_unique<RrnOrderCache>(*ranker);pl.ordering_hint=[&rrn_cache](const Board& b,const std::vector<Move>& moves){return rrn_cache->rank(b,moves);};say(std::string("info string RRN ordering enabled; learned=")+(ranker->learned()?"true":"false")+"; "+ranker->provenance());}PvsSearch pvs(pl.tt_megabytes);auto r=pvs.run(copy,pl,stop);auto ms=std::max<int64_t>(1,std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start).count());std::ostringstream info;info<<"info depth "<<r.completed_depth<<" nodes "<<r.nodes<<" time "<<ms<<" nps "<<(r.nodes*1000ULL/ms);if(std::abs(r.score_cp)>=PVS_MATE_SCORE-1000){int plies=PVS_MATE_SCORE-std::abs(r.score_cp);int mate=(plies+1)/2;info<<" score mate "<<(r.score_cp<0?-mate:mate);}else info<<" score cp "<<r.score_cp;info<<" pv";for(auto m:r.pv)info<<' '<<m.uci();say(info.str());if(rrn_cache)say("info string RRN cache hits "+std::to_string(rrn_cache->hits())+"/"+std::to_string(rrn_cache->calls()));if(!r.stop_reason.empty())say("info string search stopped: "+escape(r.stop_reason));say("bestmove "+(r.has_move?r.best.uci():"0000"));}}
                        catch(const std::exception& e){say("info string error "+escape(e.what()));say("bestmove 0000");}});
                } else if(cmd=="zero") {
                    halt();std::string op;in>>op;
                    if(op=="inspect"||op=="eval"||op=="evallegal")say(inspect(board,network,op!="inspect",op=="evallegal"));
                    else if(op=="search") {SearchLimits limits=defaults;if(!(in>>limits.simulations>>limits.seed>>limits.noise>>limits.temperature)||limits.simulations<1||limits.simulations>10000000||!std::isfinite(limits.noise)||limits.noise<0||limits.noise>1||!std::isfinite(limits.temperature)||limits.temperature<0)throw std::runtime_error("zero search SIMULATIONS SEED NOISE TEMPERATURE");stop.store(false);say(result_json(search(board,network,limits,stop)));}
                    else throw std::runtime_error("unknown zero command");
                } else if(cmd=="d")say("info string "+board.fen());
            } catch(const std::exception& e) {if(cmd=="zero")say("{\"error\":\""+escape(e.what())+"\"}");else {say("info string error "+escape(e.what()));if(cmd=="go")say("bestmove 0000");}}
        }
        halt();return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}



