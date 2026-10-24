#include "zero/verification.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
namespace zero {
float budget_importance(const std::vector<float>& priors,float value) {
    if(priors.size()<2)return .5f;
    double entropy=0;for(float p:priors)if(p>0)entropy-=p*std::log(p);
    const float uncertainty=1-std::min(1.f,std::abs(value));
    return std::clamp(float(.5+.6*entropy/std::log(double(priors.size()))+.3*uncertainty),.5f,1.4f);
}
namespace {
constexpr int piece_value[]={0,100,320,330,500,900,0};
bool capture(const Board& b,Move m) {
    return b.squares[m.to]!=0||(std::abs(b.squares[m.from])==1&&m.to==b.ep);
}
struct Verifier {
    const Network& net;const std::atomic<bool>& stop;
    std::chrono::steady_clock::time_point deadline;
    int budget,nodes=0,evals=0;bool complete=true,forcing,tactical;
    Network::Workspace workspace;
    bool expired() {
        if(nodes>=budget||stop.load(std::memory_order_relaxed)||std::chrono::steady_clock::now()>=deadline){complete=false;return true;}
        return false;
    }
    float static_value(const Board& b,const std::vector<Move>& moves) {
        ++evals;auto e=net.evaluate_legal(b,moves,workspace);float nn=e.wdl[0]-e.wdl[2];
        if(!tactical)return nn;
        int material=0;for(int p:b.squares)material+=(p>0?1:-1)*piece_value[std::abs(p)];
        // Bounded heuristic, never a mate proof. Material supplies a tactical
        // cross-check for the small WDL network, without altering its weights.
        return .25f*nn+.7f*std::tanh(float(material*b.turn)/600.f);
    }
    float quiet(const Board& b,int depth,float alpha=-1.1f,float beta=1.1f) {
        if(expired())return 0;
        ++nodes;auto moves=b.legal();float terminal;
        if(b.terminal(moves,terminal))return terminal;
        const bool checked=b.check(b.turn);
        float best=static_value(b,moves);
        if(depth<=0)return best;
        if(!checked){if(best>=beta)return best;alpha=std::max(alpha,best);}
        else best=-1.1f; // no stand pat while in check
        std::stable_sort(moves.begin(),moves.end(),[&](Move a,Move c){
            auto score=[&](Move m){return 16*piece_value[std::abs(b.squares[m.to])]-piece_value[std::abs(b.squares[m.from])]+1000*m.promo;};
            return score(a)>score(c);
        });
        for(auto move:moves) {
            Board child=b;child.make(move);
            if(!checked&&!capture(b,move)&&!move.promo&&!(forcing&&child.check(child.turn)))continue;
            float score=-quiet(child,depth-1,-beta,-alpha);
            if(!complete)return 0;
            best=std::max(best,score);alpha=std::max(alpha,score);if(alpha>=beta)break;
        }
        return best;
    }
};
}
VerificationResult verify_candidate(const Board& b,Move move,const Network& net,int depth,int nodes,
    bool counterexample,bool forcing,bool tactical,const std::atomic<bool>& stop,
    std::chrono::steady_clock::time_point deadline) {
    Verifier v{net,stop,deadline,nodes,0,0,true,forcing,tactical,{}};
    Board child=b;child.make(move);float value=0;
    if(counterexample) {
        if(v.expired())return {};
        ++v.nodes;auto replies=child.legal();float terminal;
        if(child.terminal(replies,terminal))value=-terminal;
        else {
            // A candidate must survive EVERY legal opponent reply. A truncated
            // reply set is explicitly incomplete and cannot change the move.
            value=1.1f;
            for(auto reply:replies) {
                Board reply_board=child;reply_board.make(reply);
                float score=v.quiet(reply_board,(forcing||tactical)?std::max(0,depth-2):0);
                if(!v.complete)break;
                value=std::min(value,score);
            }
        }
    } else value=-v.quiet(child,depth-1);
    return {v.complete,value,v.nodes,v.evals};
}
void verify_candidates(const Board& b,const Network& net,const SearchLimits& limits,const std::atomic<bool>& stop,
    std::chrono::steady_clock::time_point deadline,SearchResult& result) {
    if(!result.has_move||result.mate_proven||result.visits.empty())return;
    std::vector<size_t> order(result.visits.size());std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](size_t a,size_t c){
        if(result.visits[a].first==result.best)return result.visits[c].first!=result.best;
        if(result.visits[c].first==result.best)return false;
        return result.visits[a].second!=result.visits[c].second?result.visits[a].second>result.visits[c].second:result.priors[a]>result.priors[c];
    });
    if(limits.forcing_search||limits.tactical_verify) {
        // Low-prior forcing moves must have an opportunity to be checked. Keep
        // the MCTS choice, then admit ordered tactical alternatives before quiet
        // visit leaders. Scan only under the same hard time/node budget.
        std::vector<std::pair<int,size_t>> forcing_moves;
        for(size_t i=0;i<result.visits.size();++i) {
            if(stop.load(std::memory_order_relaxed)||std::chrono::steady_clock::now()>=deadline||result.verification_nodes>=limits.verification_nodes)return;
            Move m=result.visits[i].first;Board child=b;child.make(m);
            ++result.verification_nodes;
            auto replies=child.legal();float terminal=0;
            if(child.terminal(replies,terminal)&&terminal==-1) {
                // An actual terminal board is an exact one-ply proof. Visits/Q
                // remain the real MCTS statistics, including a zero-visit mate.
                result.verification_changed=m!=result.best;result.best=m;
                result.mate_proven=true;result.proof_plies=1;result.value=1;return;
            }
            const bool is_capture=capture(b,m),gives_check=child.check(child.turn);
            int score=(is_capture?16*piece_value[std::abs(b.squares[m.to])]-piece_value[std::abs(b.squares[m.from])]+100:0)
                      +(m.promo?2000+piece_value[m.promo]:0)+(gives_check&&limits.forcing_search?50:0);
            if(is_capture||m.promo||(gives_check&&limits.forcing_search))forcing_moves.push_back({score,i});
        }
        std::stable_sort(forcing_moves.begin(),forcing_moves.end(),[](auto a,auto c){return a.first>c.first;});
        std::vector<size_t> shortlist{order.front()};
        for(auto [score,index]:forcing_moves)if(index!=order.front())shortlist.push_back(index);
        for(size_t index:order)if(std::find(shortlist.begin(),shortlist.end(),index)==shortlist.end())shortlist.push_back(index);
        order=std::move(shortlist);
    }
    order.resize(std::min(order.size(),size_t(limits.verification_candidates)));
    const int quota=(limits.verification_nodes-result.verification_nodes)/std::max(1,int(order.size()));
    std::vector<float> scores;
    for(size_t index:order) {
        auto checked=verify_candidate(b,result.visits[index].first,net,limits.verification_depth,quota,
            limits.counterexample,limits.forcing_search,limits.tactical_verify,stop,deadline);
        result.verification_nodes+=checked.nodes;result.verification_evals+=checked.neural_evals;
        if(!checked.complete)return; // preserve MCTS decision on incomplete comparison
        ++result.verified_candidates;scores.push_back(checked.value);
    }
    size_t best=0;
    for(size_t i=1;i<scores.size();++i)if(scores[i]>scores[best]+1e-5f)best=i;
    if(!scores.empty()) {
        const Move choice=result.visits[order[best]].first;
        result.verification_changed=choice!=result.best;result.best=choice;
    }
}
}
