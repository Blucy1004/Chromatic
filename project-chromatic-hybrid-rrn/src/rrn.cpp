#include "zero/rrn.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <utility>
#include <string_view>

namespace zero {
namespace {
constexpr std::string_view MAGIC="CHROMATIC_RRN_WEIGHTS";
constexpr int FORMAT_VERSION=1;
constexpr int NODE_COUNT=64;
using Hidden=std::array<float,RRN_HIDDEN>;
using Features=std::array<float,RRN_FEATURES>;
using Matrix=std::array<float,RRN_HIDDEN*RRN_HIDDEN>;

bool finite(float value) { return std::isfinite(value); }
void validate(const RrnWeights& w) {
    if(w.provenance.empty() || w.provenance.size()>128 || w.provenance.find_first_of(" \t\r\n")!=std::string::npos)
        throw std::runtime_error("RRN provenance must be a nonempty token of at most 128 bytes");
    auto check=[](const auto& values){for(float x:values)if(!finite(x)||std::abs(x)>10000.0f)throw std::runtime_error("RRN weight is non-finite or outside the supported magnitude");};
    check(w.input);check(w.input_bias);check(w.self);check(w.relation);check(w.message_bias);check(w.rank);
    if(!finite(w.rank_bias)||std::abs(w.rank_bias)>10000.0f)throw std::runtime_error("RRN rank bias is non-finite or outside the supported magnitude");
}
uint32_t next_rng(uint32_t& x) {
    // xorshift32 is used only to make a reproducible reference fixture, not training.
    x^=x<<13;x^=x>>17;x^=x<<5;return x;
}
float small_weight(uint32_t& state,float scale) {
    const int centered=static_cast<int>(next_rng(state)&0xffffu)-32768;
    return (static_cast<float>(centered)/32768.0f)*scale;
}
int piece_type(int p) {return std::abs(p);}
Features node_features(const Board& board,int square) {
    Features x{};const int p=board.squares[square];
    if(p!=0)x[(p>0?0:6)+piece_type(p)-1]=1.0f;
    x[12]=board.turn==1?1.0f:0.0f;
    for(int r=0;r<4;++r)x[13+r]=(board.castle&(1<<r))?1.0f:0.0f;
    x[17]=board.ep==square?1.0f:0.0f;
    x[18]=std::clamp(static_cast<float>(board.halfmove)/100.0f,0.0f,1.0f);
    return x;
}
RrnNode make_node(const Board& board,int square) {
    const int p=board.squares[square];
    return {square,p?static_cast<RrnNodeType>(piece_type(p)):RrnNodeType::EmptySquare,p>0?1:(p<0?-1:0)};
}
void matvec(const Matrix& matrix,const Hidden& in,Hidden& out) {
    for(int o=0;o<RRN_HIDDEN;++o){float v=0;for(int i=0;i<RRN_HIDDEN;++i)v+=matrix[o*RRN_HIDDEN+i]*in[i];out[o]=v;}
}
void add_matvec(const Matrix& matrix,const Hidden& in,Hidden& out) {
    for(int o=0;o<RRN_HIDDEN;++o){float v=0;for(int i=0;i<RRN_HIDDEN;++i)v+=matrix[o*RRN_HIDDEN+i]*in[i];out[o]+=v;}
}
Matrix relation_matrix(const RrnWeights& w,int relation) {
    Matrix m{};const size_t offset=static_cast<size_t>(relation)*RRN_HIDDEN*RRN_HIDDEN;
    std::copy_n(w.relation.begin()+offset,m.size(),m.begin());return m;
}
std::array<float,5> move_features(const Board& board,const Move& move) {
    const int mover=piece_type(board.squares[move.from]);
    int captured=piece_type(board.squares[move.to]);
    if(mover==1 && move.to==board.ep && board.squares[move.to]==0 && move.from%8!=move.to%8)captured=1;
    const float target_center=1.0f-(std::abs(3-(move.to%8))+std::abs(3-(move.to/8)))/6.0f;
    const bool castle=mover==6&&std::abs(move.to-move.from)==2;
    return {mover/6.0f,captured/5.0f,move.promo?move.promo/5.0f:0.0f,target_center,castle?1.0f:0.0f};
}
}

RrnWeights RrnWeights::deterministic_reference(uint32_t seed) {
    RrnWeights w;w.seed=seed;w.learned=false;w.provenance="deterministic-reference-v1";
    uint32_t state=seed?seed:0x9e3779b9u;
    for(float& x:w.input)x=small_weight(state,0.08f);
    for(float& x:w.input_bias)x=small_weight(state,0.015f);
    for(int i=0;i<RRN_HIDDEN;++i)w.self[i*RRN_HIDDEN+i]=0.20f;
    for(int r=0;r<RRN_RELATIONS;++r){
        const size_t base=static_cast<size_t>(r)*RRN_HIDDEN*RRN_HIDDEN;
        for(int i=0;i<RRN_HIDDEN;++i)w.relation[base+i*RRN_HIDDEN+i]=0.025f*(r+1);
        for(int i=0;i<RRN_HIDDEN*RRN_HIDDEN;++i)w.relation[base+i]+=small_weight(state,0.006f);
    }
    for(float& x:w.message_bias)x=small_weight(state,0.01f);
    for(float& x:w.rank)x=small_weight(state,0.05f);
    w.rank_bias=0.0f;validate(w);return w;
}

void RrnWeights::save(const std::string& path) const {
    validate(*this);std::ofstream out(path,std::ios::binary|std::ios::trunc);
    if(!out)throw std::runtime_error("cannot write RRN weights: "+path);
    out<<MAGIC<<' '<<FORMAT_VERSION<<' '<<RRN_FEATURES<<' '<<RRN_HIDDEN<<' '<<RRN_RELATIONS<<' '<<RRN_MESSAGE_STEPS<<'\n';
    out<<"learned "<<(learned?1:0)<<" seed "<<seed<<" provenance "<<provenance<<'\n';
    out<<std::setprecision(std::numeric_limits<float>::max_digits10);
    auto emit=[&](std::string_view name,const auto& a){out<<name;for(float x:a)out<<' '<<x;out<<'\n';};
    emit("input",input);emit("input_bias",input_bias);emit("self",self);emit("relation",relation);emit("message_bias",message_bias);emit("rank",rank);out<<"rank_bias "<<rank_bias<<'\n';
    if(!out)throw std::runtime_error("failed while writing RRN weights: "+path);
}

RrnWeights RrnWeights::load(const std::string& path) {
    std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("cannot open RRN weights: "+path);
    std::string magic;int version=0,features=0,hidden=0,relations=0,steps=0;
    if(!(in>>magic>>version>>features>>hidden>>relations>>steps)||magic!=MAGIC||version!=FORMAT_VERSION||features!=RRN_FEATURES||hidden!=RRN_HIDDEN||relations!=RRN_RELATIONS||steps!=RRN_MESSAGE_STEPS)
        throw std::runtime_error("unsupported RRN weights header");
    RrnWeights w;int learned_value=0;std::string learned_label,seed_label,provenance_label;
    if(!(in>>learned_label>>learned_value>>seed_label>>w.seed>>provenance_label>>w.provenance)||learned_label!="learned"||seed_label!="seed"||provenance_label!="provenance"||(learned_value!=0&&learned_value!=1))
        throw std::runtime_error("invalid RRN weights metadata");
    w.learned=learned_value!=0;
    auto read=[&](std::string_view expected,auto& a){std::string key;if(!(in>>key)||key!=expected)throw std::runtime_error("invalid RRN weights section: "+std::string(expected));for(float& x:a)if(!(in>>x)||!finite(x))throw std::runtime_error("invalid/non-finite RRN weight in "+std::string(expected));};
    read("input",w.input);read("input_bias",w.input_bias);read("self",w.self);read("relation",w.relation);read("message_bias",w.message_bias);read("rank",w.rank);
    std::string rank_label;if(!(in>>rank_label>>w.rank_bias)||rank_label!="rank_bias"||!finite(w.rank_bias))throw std::runtime_error("invalid RRN rank bias");
    std::string extra;if(in>>extra)throw std::runtime_error("trailing data in RRN weights");validate(w);return w;
}

RrnRanker::RrnRanker(RrnWeights weights):weights_(std::move(weights)){validate(weights_);}

std::vector<float> RrnRanker::rank(const Board& board,const std::vector<Move>& legal_moves) const {
    if(legal_moves.empty())return {};
    std::array<RrnNode,NODE_COUNT> nodes{};
    std::array<Features,NODE_COUNT> x{};
    std::array<Hidden,NODE_COUNT> h{},next{};
    for(int s=0;s<NODE_COUNT;++s){nodes[s]=make_node(board,s);x[s]=node_features(board,s);for(int o=0;o<RRN_HIDDEN;++o){float v=weights_.input_bias[o];for(int f=0;f<RRN_FEATURES;++f)v+=x[s][f]*weights_.input[f*RRN_HIDDEN+o];h[s][o]=std::tanh(v);}}

    std::vector<RrnEdge> edges;edges.reserve(700);
    auto push=[&](int source,int target,RrnRelationType type){edges.push_back({static_cast<uint8_t>(source),static_cast<uint8_t>(target),type});};
    for(int a=0;a<NODE_COUNT;++a)for(int b=0;b<NODE_COUNT;++b){if(a==b)continue;
        const int dx=std::abs(a%8-b%8),dy=std::abs(a/8-b/8);
        if(dx<=1&&dy<=1)push(a,b,RrnRelationType::AdjacentSquare);
        if(nodes[a].color&&nodes[a].color==nodes[b].color)push(a,b,RrnRelationType::SameColor);
        if(nodes[a].type!=RrnNodeType::EmptySquare&&nodes[a].type==nodes[b].type&&nodes[a].color==nodes[b].color)push(a,b,RrnRelationType::SamePieceType);
    }
    for(const Move& m:legal_moves){if(m.from<0||m.from>=64||m.to<0||m.to>=64)throw std::runtime_error("RRN received invalid move square");push(m.from,m.to,RrnRelationType::LegalCandidate);}

    std::array<Matrix,RRN_RELATIONS> rel{};for(int r=0;r<RRN_RELATIONS;++r)rel[r]=relation_matrix(weights_,r);
    for(int step=0;step<RRN_MESSAGE_STEPS;++step){
        std::array<std::array<Hidden,RRN_RELATIONS>,NODE_COUNT> aggregate{};
        std::array<std::array<int,RRN_RELATIONS>,NODE_COUNT> counts{};
        for(const auto& e:edges){const int r=static_cast<int>(e.relation);auto& sum=aggregate[e.target][r];const auto& src=h[e.source];for(int k=0;k<RRN_HIDDEN;++k)sum[k]+=src[k];++counts[e.target][r];}
        for(int target=0;target<NODE_COUNT;++target){
            Hidden z{};matvec(weights_.self,h[target],z);
            for(int k=0;k<RRN_HIDDEN;++k)z[k]+=weights_.message_bias[k];
            for(int r=0;r<RRN_RELATIONS;++r){const int n=counts[target][r];if(!n)continue;Hidden mean=aggregate[target][r];for(float& v:mean)v/=static_cast<float>(n);add_matvec(rel[r],mean,z);}
            for(int k=0;k<RRN_HIDDEN;++k)next[target][k]=std::tanh(z[k]);
        }
        h.swap(next);
    }

    std::vector<float> scores;scores.reserve(legal_moves.size());
    for(const Move& m:legal_moves){const auto mf=move_features(board,m);float score=weights_.rank_bias;int index=0;
        for(int k=0;k<RRN_HIDDEN;++k)score+=weights_.rank[index++]*h[m.from][k];
        for(int k=0;k<RRN_HIDDEN;++k)score+=weights_.rank[index++]*h[m.to][k];
        for(float v:mf)score+=weights_.rank[index++]*v;
        scores.push_back(score);
    }
    return scores;
}

RrnOrderCache::RrnOrderCache(const RrnRanker& ranker,size_t capacity)
    :ranker_(ranker),entries_(std::max<size_t>(1,capacity)) {}

std::vector<float> RrnOrderCache::rank(const Board& board,const std::vector<Move>& legal_moves) {
    ++calls_;
    uint64_t hash=14695981039346656037ULL;
    auto mix=[&](uint64_t value){hash^=value;hash*=1099511628211ULL;};
    for(int piece:board.squares)mix(static_cast<uint64_t>(piece+6));
    mix(static_cast<uint64_t>(board.turn+1));mix(static_cast<uint64_t>(board.castle));
    mix(static_cast<uint64_t>(board.ep+1));mix(static_cast<uint64_t>(std::max(0,board.halfmove)));
    for(const Move& move:legal_moves){mix(static_cast<uint64_t>(move.from));mix(static_cast<uint64_t>(move.to));mix(static_cast<uint64_t>(move.promo));}
    Entry& entry=entries_[static_cast<size_t>(hash%entries_.size())];
    if(entry.valid&&entry.hash==hash&&entry.squares==board.squares&&entry.turn==board.turn&&
       entry.castle==board.castle&&entry.ep==board.ep&&entry.halfmove==board.halfmove&&entry.moves==legal_moves) {
        ++hits_;return entry.scores;
    }
    auto scores=ranker_.rank(board,legal_moves);
    entry.valid=true;entry.hash=hash;entry.squares=board.squares;entry.turn=board.turn;
    entry.castle=board.castle;entry.ep=board.ep;entry.halfmove=board.halfmove;
    entry.moves=legal_moves;entry.scores=scores;
    return scores;
}
}



