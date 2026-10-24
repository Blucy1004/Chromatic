#include "zero/search.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <thread>

using namespace zero;
namespace {
std::mutex output_mutex;
void say(const std::string& s){std::lock_guard lock(output_mutex);std::cout<<s<<std::endl;}
std::string escape(const std::string& s){std::string r;for(unsigned char c:s){if(c=='"'||c=='\\')r+='\\';if(c>=32)r+=char(c);}return r;}
template<class T>void array(std::ostream& o,const T& values){o<<'[';bool first=true;for(auto v:values){if(!first)o<<',';first=false;o<<v;}o<<']';}
std::string inspect(const Board& b,const Network& net,bool evaluation){
    std::ostringstream o;o<<std::setprecision(9);auto moves=b.legal();float value=0;bool terminal=b.terminal(moves,value);
    o<<"{\"fen\":\""<<b.fen()<<"\",\"repetitions\":"<<b.repetitions()<<",\"terminal\":"<<(terminal?"true":"false")<<",\"value\":"<<value<<",\"features\":";array(o,b.features());
    o<<",\"legal\":[";bool first=true;for(auto m:moves){if(!first)o<<',';first=false;o<<"{\"uci\":\""<<m.uci()<<"\",\"action\":"<<action(m)<<'}';}o<<']';
    if(evaluation){auto e=net.evaluate(b);o<<",\"policy\":";array(o,e.policy);o<<",\"wdl\":";array(o,e.wdl);}
    o<<'}';return o.str();
}
std::string result_json(const SearchResult& r){
    std::ostringstream o;o<<std::setprecision(9)<<"{\"bestmove\":\""<<(r.has_move?r.best.uci():"0000")<<"\",\"terminal\":"<<(r.terminal?"true":"false")<<",\"simulations\":"<<r.simulations<<",\"value\":"<<r.value<<",\"visits\":[";
    bool first=true;for(auto& [m,n]:r.visits){if(!first)o<<',';first=false;o<<"{\"uci\":\""<<m.uci()<<"\",\"action\":"<<action(m)<<",\"n\":"<<n<<'}';}o<<"]}";return o.str();
}
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
        Board board=Board::from_fen(START_FEN);Network network;if(!model_path.empty())network.load(model_path);
        SearchLimits defaults;std::atomic<bool> stop{false};std::thread worker;
        auto halt=[&](){stop.store(true);if(worker.joinable())worker.join();};
        std::string line;
        while(std::getline(std::cin,line)) {
            std::istringstream in(line);std::string cmd;in>>cmd;
            try {
                if(cmd=="quit"){halt();break;}
                if(cmd=="uci") {say("id name PROJECT ZERO 0.1\nid author PROJECT ZERO\noption name ModelFile type string default "+(model_path.empty()?"<none>":model_path)+"\noption name CPuct type string default 1.5\noption name Simulations type spin default 800 min 1 max 10000000\noption name MaxNodes type spin default 100000 min 16 max 1000000\nuciok");}
                else if(cmd=="isready")say("readyok");
                else if(cmd=="stop")halt();
                else if(cmd=="ucinewgame"){halt();board=Board::from_fen(START_FEN);}
                else if(cmd=="position"){halt();board=position(in);}
                else if(cmd=="setoption") {
                    halt();std::string token,name,value;in>>token;if(token!="name")throw std::runtime_error("missing option name");
                    while(in>>token&&token!="value"){if(!name.empty())name+=' ';name+=token;}
                    std::getline(in,value);if(!value.empty()&&value.front()==' ')value.erase(0,1);
                    if(name=="ModelFile"){if(value=="<none>"||value.empty()){network=Network{};model_path.clear();}else {network.load(value);model_path=value;}say("info string model "+std::string(network.loaded()?"loaded":"uniform baseline"));}
                    else if(name=="CPuct"){float v=std::stof(value);if(!std::isfinite(v)||v<=0||v>10)throw std::runtime_error("CPuct must be in (0,10]");defaults.cpuct=v;}
                    else if(name=="Simulations")defaults.simulations=std::clamp(std::stoi(value),1,10000000);
                    else if(name=="MaxNodes")defaults.max_nodes=size_t(std::clamp(std::stoi(value),16,1000000));
                } else if(cmd=="go") {
                    halt();SearchLimits limits=defaults;std::string token;int remaining=0,increment=0,moves_to_go=30;bool explicit_nodes=false,explicit_time=false,infinite=false,has_clock=false;
                    while(in>>token){if(token=="infinite"){infinite=true;continue;}if(token=="ponder"||token=="searchmoves")throw std::runtime_error("unsupported go option: "+token);long long n;if(!(in>>n))throw std::runtime_error("missing go limit");n=std::clamp(n,0LL,1000000000LL);
                        if(token=="nodes"){limits.simulations=int(std::max(1LL,n));explicit_nodes=true;}
                        else if(token=="movetime"){limits.milliseconds=int(std::max(1LL,n));explicit_time=true;}
                        else if(token==(board.turn==1?"wtime":"btime")){remaining=int(n);has_clock=true;}
                        else if(token==(board.turn==1?"winc":"binc"))increment=int(n);
                        else if(token=="movestogo")moves_to_go=std::max(1,int(n));
                        else if(token=="depth") {limits.simulations=int(std::clamp(n*100,1LL,10000000LL));explicit_nodes=true;}
                    }
                    if(!explicit_time&&has_clock)limits.milliseconds=std::max(1,std::min(remaining/2,remaining/moves_to_go+increment/2));
                    if(infinite){limits.simulations=0;limits.milliseconds=0;}else if(limits.milliseconds>0&&!explicit_nodes)limits.simulations=0;
                    stop.store(false);auto copy=board;
                    worker=std::thread([&,copy,limits](){try {auto start=std::chrono::steady_clock::now();auto r=search(copy,network,limits,stop);auto ms=std::max<int64_t>(1,std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-start).count());say("info nodes "+std::to_string(r.simulations)+" time "+std::to_string(ms)+" nps "+std::to_string(r.simulations*1000LL/ms)+" score cp "+std::to_string(int(400*r.value))+" pv "+(r.has_move?r.best.uci():"0000"));say("bestmove "+(r.has_move?r.best.uci():"0000"));}catch(const std::exception& e){say("info string error "+escape(e.what()));say("bestmove 0000");}});
                } else if(cmd=="zero") {
                    halt();std::string op;in>>op;
                    if(op=="inspect"||op=="eval")say(inspect(board,network,op=="eval"));
                    else if(op=="search") {SearchLimits limits=defaults;if(!(in>>limits.simulations>>limits.seed>>limits.noise>>limits.temperature)||limits.simulations<1||limits.simulations>10000000||!std::isfinite(limits.noise)||limits.noise<0||limits.noise>1||!std::isfinite(limits.temperature)||limits.temperature<0)throw std::runtime_error("zero search SIMULATIONS SEED NOISE TEMPERATURE");stop.store(false);say(result_json(search(board,network,limits,stop)));}
                    else throw std::runtime_error("unknown zero command");
                } else if(cmd=="d")say("info string "+board.fen());
            } catch(const std::exception& e) {if(cmd=="zero")say("{\"error\":\""+escape(e.what())+"\"}");else say("info string error "+escape(e.what()));}
        }
        halt();return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
