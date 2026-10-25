#include "zero/pvs.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>

namespace zero {
namespace {
constexpr int INF=32000;
constexpr int MATE_THRESHOLD=PVS_MATE_SCORE-512;
constexpr int MAX_PLY=256;
constexpr std::array<int,7> MATERIAL={0,100,320,335,500,900,0};

enum Bound:uint8_t { EXACT=0, LOWER=1, UPPER=2 };
struct Entry {
    uint64_t key=0,history_context=0;
    uint32_t halfmove=0;
    int16_t score=0;
    int8_t depth=-1;
    uint8_t bound=UPPER;
    Move best{};
    bool valid=false;
};

int evaluate_white(const Board& b) {
    int score=0,white_bishops=0,black_bishops=0;
    for(int s=0;s<64;++s) {
        const int p=b.squares[s];if(!p)continue;
        const int type=std::abs(p),color=p>0?1:-1;
        // Use a fixed White perspective for evaluation, then flip to stm.
        const int rank=s/8,file=s%8;
        const int relative_rank=color==1?rank:7-rank;
        const int center_file=7-std::abs(2*file-7);
        const int center_rank=7-std::abs(2*rank-7);
        const int center=center_file+center_rank;
        int bonus=0;
        switch(type) {
            case 1: bonus=relative_rank*7+center_file*2;break;
            case 2: bonus=center*5-(relative_rank==0?8:0);break;
            case 3: bonus=center*3;break;
            case 4: bonus=relative_rank*2+center_file;break;
            case 5: bonus=center;break;
            case 6: bonus=-center*2+(relative_rank==0?5:0);break;
            default:break;
        }
        score+=color*(MATERIAL[type]+bonus);
        if(type==3){if(color==1)++white_bishops;else ++black_bishops;}
    }
    if(white_bishops>=2)score+=25;
    if(black_bishops>=2)score-=25;
    return score;
}
int evaluate(const Board& b) { return b.turn*evaluate_white(b); }

uint64_t history_context(const Board& b) {
    // Repetition is path-dependent. Bind TT scores to the complete reversible
    // history window so a score from a different repetition context is not used.
    uint64_t h=14695981039346656037ULL;
    for(uint64_t x:b.history){h^=x;h*=1099511628211ULL;}
    h^=uint64_t(b.history.size());h*=1099511628211ULL;
    return h;
}
int score_to_tt(int score,int ply) {
    if(score>MATE_THRESHOLD)return score+ply;
    if(score<-MATE_THRESHOLD)return score-ply;
    return score;
}
int score_from_tt(int score,int ply) {
    if(score>MATE_THRESHOLD)return score-ply;
    if(score<-MATE_THRESHOLD)return score+ply;
    return score;
}
bool same_move(const Move& a,const Move& b){return a==b;}
}

struct PvsSearch::Impl {
    std::vector<Entry> table;
    size_t allocated_mb=0;
    PvsResult result{};
    const PvsLimits* limits=nullptr;
    const std::atomic<bool>* stop=nullptr;
    std::chrono::steady_clock::time_point deadline{};
    bool timed=false,aborted=false;

    void ensure_table(size_t mb) {
        mb=std::max<size_t>(1,std::min<size_t>(mb,256));
        if(allocated_mb==mb&&!table.empty())return;
        allocated_mb=mb;
        const size_t count=std::max<size_t>(1,(mb*1024*1024)/sizeof(Entry));
        table.assign(count,Entry{});
    }
    void clear(){std::fill(table.begin(),table.end(),Entry{});}
    bool halt_requested() {
        if(aborted)return true;
        if(stop&&stop->load(std::memory_order_relaxed)){aborted=true;result.stop_reason="stop";return true;}
        if(result.nodes>=limits->max_nodes){aborted=true;result.stop_reason="nodes";return true;}
        if(timed&&std::chrono::steady_clock::now()>=deadline){aborted=true;result.stop_reason="time";return true;}
        return false;
    }
    bool enter_node(bool qnode) {
        if(halt_requested())return false;
        ++result.nodes;if(qnode)++result.qnodes;
        return true;
    }
    int terminal_score(const Board& b,const std::vector<Move>& moves,int ply,bool& terminal) {
        float v=0;
        if(!b.terminal(moves,v))return 0;
        terminal=true;
        if(moves.empty()&&b.check(b.turn))return -PVS_MATE_SCORE+ply;
        return 0;
    }
    Entry* probe(const Board& b,uint64_t ctx) {
        if(table.empty())return nullptr;
        Entry& e=table[size_t(b.key()%table.size())];
        if(!e.valid||e.key!=b.key()||e.halfmove!=uint32_t(std::max(0,b.halfmove))||e.history_context!=ctx)return nullptr;
        ++result.tt_hits;return &e;
    }
    void store(const Board& b,uint64_t ctx,int depth,int score,Bound bound,const Move& best,int ply) {
        if(table.empty())return;
        Entry& e=table[size_t(b.key()%table.size())];
        if(e.valid&&e.key==b.key()&&e.halfmove==uint32_t(std::max(0,b.halfmove))&&e.history_context==ctx&&e.depth>depth&&bound!=EXACT)return;
        if(e.valid&&e.key!=b.key()&&e.depth>depth+2)return;
        e.key=b.key();e.history_context=ctx;e.halfmove=uint32_t(std::max(0,b.halfmove));
        e.score=int16_t(std::clamp(score_to_tt(score,ply),-32767,32767));
        e.depth=int8_t(std::clamp(depth,-1,127));e.bound=uint8_t(bound);e.best=best;e.valid=true;
    }
    void order_moves(const Board& b,std::vector<Move>& moves,const Move* tt_move,int depth) {
        std::vector<float> hints;
        if(limits->ordering_hint&&(limits->ordering_hint_min_depth<=0||depth>=limits->ordering_hint_min_depth)) {
            try {hints=limits->ordering_hint(b,moves);} catch(...) {hints.clear();}
            if(hints.size()!=moves.size())hints.clear();
        }
        struct RankedMove { int category=0,tactical=0;float hint=0;Move move{}; };
        std::vector<RankedMove> ranked;ranked.reserve(moves.size());
        for(size_t i=0;i<moves.size();++i) {
            const Move& m=moves[i];
            const bool tt=tt_move&&same_move(m,*tt_move);
            const int victim=(m.to==b.ep&&std::abs(b.squares[m.from])==1&&b.squares[m.to]==0)?1:std::abs(b.squares[m.to]);
            const bool capture=victim!=0;
            // Strict tactical tiers ensure no neural/reference score can demote
            // a TT move or promote a quiet move above a capture/promotion.
            const int category=tt?4:(capture?3:(m.promo?2:1));
            const int tactical=capture?(MATERIAL[victim]*16-MATERIAL[std::abs(b.squares[m.from])]+(m.promo?MATERIAL[m.promo]:0)):(m.promo?MATERIAL[m.promo]:0);
            float hint=0.0f;
            if(!hints.empty()&&std::isfinite(hints[i]))hint=std::clamp(hints[i],-10000.0f,10000.0f);
            ranked.push_back({category,tactical,hint,m});
        }
        std::stable_sort(ranked.begin(),ranked.end(),[](const RankedMove& a,const RankedMove& b){
            if(a.category!=b.category)return a.category>b.category;
            if(a.tactical!=b.tactical)return a.tactical>b.tactical;
            // Keep fractional RRN scores: integer conversion erases small hints.
            return a.hint>b.hint;
        });
        for(size_t i=0;i<moves.size();++i)moves[i]=ranked[i].move;
    }
    int qsearch(Board& b,int alpha,int beta,int ply,int qply,std::vector<Move>& pv) {
        if(!enter_node(true))return 0;
        auto moves=b.legal();bool terminal=false;int ts=terminal_score(b,moves,ply,terminal);if(terminal)return ts;
        const bool in_check=b.check(b.turn);
        if(!in_check) {
            const int stand=evaluate(b);
            if(stand>=beta)return stand;
            if(stand>alpha)alpha=stand;
            if(qply>=16)return stand;
        }
        // In check there is no stand-pat: search every legal evasion, including
        // quiet moves. At the quiescence cap, checked nodes still resolve evasions.
        if(in_check)order_moves(b,moves,nullptr,-1);
        int best=in_check?-INF:evaluate(b);
        bool searched=false;
        for(const Move& m:moves) {
            if(!in_check) {
                const bool ep_capture=(m.to==b.ep&&std::abs(b.squares[m.from])==1&&b.squares[m.to]==0);
                if(!b.squares[m.to]&&!ep_capture&&!m.promo)continue;
            }
            if(halt_requested())break;
            Board child=b;child.make(m,true);std::vector<Move> child_pv;
            const int score=-qsearch(child,-beta,-alpha,ply+1,qply+1,child_pv);if(aborted)break;
            searched=true;
            if(score>best){best=score;pv.clear();pv.push_back(m);pv.insert(pv.end(),child_pv.begin(),child_pv.end());}
            if(score>alpha)alpha=score;
            if(alpha>=beta)break;
        }
        if(in_check&&!searched&&!aborted)return -PVS_MATE_SCORE+ply;
        return best;
    }
    int negamax(Board& b,int depth,int alpha,int beta,int ply,std::vector<Move>& pv) {
        if(!enter_node(false))return 0;
        auto moves=b.legal();bool terminal=false;int ts=terminal_score(b,moves,ply,terminal);if(terminal)return ts;
        if(depth<=0)return qsearch(b,alpha,beta,ply,0,pv);
        if(ply>=MAX_PLY&&!b.check(b.turn))return evaluate(b);
        if(ply>=MAX_PLY)depth=0; // Continue only through checked positions, without standing pat.
        const int alpha_orig=alpha,beta_orig=beta;const uint64_t ctx=history_context(b);
        Entry* hit=probe(b,ctx);Move tt_move{};bool has_tt_move=false;
        if(hit){tt_move=hit->best;has_tt_move=true;const int v=score_from_tt(hit->score,ply);
            if(ply>0&&hit->depth>=depth){if(hit->bound==EXACT){++result.tt_cutoffs;pv={hit->best};return v;}if(hit->bound==LOWER&&v>=beta){++result.tt_cutoffs;pv={hit->best};return v;}if(hit->bound==UPPER&&v<=alpha){++result.tt_cutoffs;pv={hit->best};return v;}}
        }
        order_moves(b,moves,has_tt_move?&tt_move:nullptr,depth);
        int best=-INF;Move best_move=moves.front();bool first=true;
        for(const Move& m:moves) {
            if(halt_requested())break;
            Board child=b;child.make(m,true);std::vector<Move> child_pv;int score;
            if(first)score=-negamax(child,depth-1,-beta,-alpha,ply+1,child_pv);
            else {
                score=-negamax(child,depth-1,-alpha-1,-alpha,ply+1,child_pv);
                if(!aborted&&score>alpha&&score<beta){child_pv.clear();score=-negamax(child,depth-1,-beta,-alpha,ply+1,child_pv);}
            }
            if(aborted)break;
            first=false;
            if(score>best){best=score;best_move=m;pv.clear();pv.push_back(m);pv.insert(pv.end(),child_pv.begin(),child_pv.end());}
            if(score>alpha)alpha=score;
            if(alpha>=beta)break;
        }
        if(aborted)return 0;
        const Bound bound=best<=alpha_orig?UPPER:(best>=beta_orig?LOWER:EXACT);
        store(b,ctx,depth,best,bound,best_move,ply);
        return best;
    }
};

PvsSearch::PvsSearch(size_t mb):impl_(new Impl){impl_->ensure_table(mb);}
PvsSearch::~PvsSearch(){delete impl_;}
PvsSearch::PvsSearch(PvsSearch&& x) noexcept:impl_(x.impl_){x.impl_=nullptr;}
PvsSearch& PvsSearch::operator=(PvsSearch&& x) noexcept {if(this!=&x){delete impl_;impl_=x.impl_;x.impl_=nullptr;}return *this;}
void PvsSearch::clear(){if(impl_)impl_->clear();}
PvsResult PvsSearch::run(const Board& root,const PvsLimits& limits,const std::atomic<bool>& stop) {
    if(!impl_)impl_=new Impl;
    impl_->ensure_table(limits.tt_megabytes);impl_->result={};impl_->limits=&limits;impl_->stop=&stop;impl_->aborted=false;impl_->timed=limits.milliseconds>0;
    const auto start=std::chrono::steady_clock::now();impl_->deadline=start+std::chrono::milliseconds(std::max(0,limits.milliseconds));
    PvsResult& out=impl_->result;Board board=root;auto moves=board.legal();
    float terminal_value=0;
    if(board.terminal(moves,terminal_value)){out.terminal=true;out.completed=true;out.stop_reason="terminal";if(moves.empty()&&board.check(board.turn))out.score_cp=-PVS_MATE_SCORE;return out;}
    out.has_move=true;out.best=moves.front();
    if(limits.max_depth<=0){out.stop_reason="depth";return out;}
    for(int depth=1;depth<=limits.max_depth;++depth) {
        std::vector<Move> pv;int score=impl_->negamax(board,depth,-INF,INF,0,pv);
        if(impl_->aborted)break;
        if(pv.empty())break;
        out.best=pv.front();out.pv=std::move(pv);out.score_cp=score;out.completed_depth=depth;out.completed=true;
        if(std::abs(score)>MATE_THRESHOLD)break;
    }
    if(impl_->aborted)return out;
    if(out.stop_reason.empty())out.stop_reason=out.completed_depth>=limits.max_depth?"depth":"terminal";
    return out;
}
}
