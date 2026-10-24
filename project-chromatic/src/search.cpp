#include "zero/search.hpp"
#include "zero/proof.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace zero {
namespace {
constexpr uint32_t NONE=std::numeric_limits<uint32_t>::max();
struct Edge {
    Move move;
    float prior=0;
    double sum=0;
    uint32_t visits=0,child=NONE;
};
struct Node {
    uint32_t first=0,count=0,visits=0;
    float value=0;
    bool expanded=false,terminal=false;
};
// Integer indices remain valid when vectors grow. No pointers outlive a reserve.
class Arena {
public:
    std::vector<Node> nodes;
    std::vector<Edge> edges;
    const SearchLimits& limits;
    explicit Arena(const SearchLimits& l):limits(l) {
        if(l.max_nodes<1||l.max_tree_bytes<65536)throw std::runtime_error("invalid tree memory budget");
        nodes.reserve(std::min<size_t>(256,l.max_nodes));nodes.emplace_back();
        edges.reserve(std::min<size_t>(1024,(l.max_tree_bytes-nodes.capacity()*sizeof(Node))/sizeof(Edge)));
    }
    size_t bytes() const{return nodes.capacity()*sizeof(Node)+edges.capacity()*sizeof(Edge);}
    uint32_t child(uint32_t edge) {
        if(edges[edge].child!=NONE)return edges[edge].child;
        if(nodes.size()>=limits.max_nodes)return NONE;
        if(nodes.size()==nodes.capacity()) {
            size_t available=(limits.max_tree_bytes-edges.capacity()*sizeof(Edge))/sizeof(Node);
            size_t capacity=std::min({limits.max_nodes,available,std::max<size_t>(64,nodes.capacity()*2)});
            if(capacity<=nodes.size())return NONE;
            nodes.reserve(capacity);
        }
        auto index=uint32_t(nodes.size());nodes.emplace_back();edges[edge].child=index;return index;
    }
    bool children(uint32_t index,const std::vector<Move>& moves,const Evaluation& eval) {
        size_t needed=edges.size()+moves.size();
        if(needed>edges.capacity()) {
            size_t available=(limits.max_tree_bytes-nodes.capacity()*sizeof(Node))/sizeof(Edge);
            size_t capacity=std::min(available,std::max(needed,edges.capacity()*2));
            if(capacity<needed)return false;
            edges.reserve(capacity);
        }
        float maximum=-std::numeric_limits<float>::infinity(),sum=0;
        for(auto m:moves)maximum=std::max(maximum,eval.policy[action(m)]);
        uint32_t first=uint32_t(edges.size());
        for(auto m:moves){float p=std::exp(eval.policy[action(m)]-maximum);sum+=p;edges.push_back({m,p,0,0,NONE});}
        for(size_t i=first;i<edges.size();++i)edges[i].prior/=sum;
        nodes[index].first=first;nodes[index].count=uint32_t(moves.size());nodes[index].expanded=true;
        return true;
    }
};
float evaluate(Arena& tree,uint32_t index,const Board& b,const Network& net,SearchResult& result,Network::Workspace& workspace) {
    auto moves=b.legal();float value;
    if(b.terminal(moves,value)) {
        if(index!=NONE){auto& n=tree.nodes[index];n.expanded=n.terminal=true;n.value=value;}
        return value;
    }
    ++result.neural_evals;auto eval=net.evaluate_legal(b,moves,workspace);value=eval.wdl[0]-eval.wdl[2];
    if(index!=NONE){tree.nodes[index].value=value;tree.children(index,moves,eval);}
    return value;
}
}

SearchResult search(const Board& board,const Network& network,const SearchLimits& limits,const std::atomic<bool>& stop) {
    const auto start=std::chrono::steady_clock::now();
    auto cancelled=[&](){return stop.load(std::memory_order_relaxed)||(limits.milliseconds>0&&std::chrono::steady_clock::now()-start>=std::chrono::milliseconds(limits.milliseconds));};
    Arena tree(limits);SearchResult result;Network::Workspace workspace;
    result.value=evaluate(tree,0,board,network,result,workspace);result.terminal=tree.nodes[0].terminal;
    result.arena_nodes=tree.nodes.size();result.memory_bytes=tree.bytes();
    if(result.terminal)return result;
    if(!tree.nodes[0].expanded)throw std::runtime_error("tree budget cannot hold root");
    std::mt19937_64 rng(limits.seed);
    const uint32_t root_count=tree.nodes[0].count;
    if(limits.proof_nodes>0&&!cancelled()) {
        std::vector<Move> moves;for(uint32_t i=0;i<root_count;++i)moves.push_back(tree.edges[i].move);
        int microseconds=limits.milliseconds>0?std::max(100, std::min(10000,limits.milliseconds*1000/4)):0;
        auto proof=verify_mate(board,moves,limits.proof_plies,limits.proof_nodes,stop,microseconds);
        result.proof_nodes=proof.nodes;
        if(proof.proven) {
            // One actual root update backed by a proof; do not invent requested visits.
            result.best=proof.move;result.has_move=result.mate_proven=true;result.value=1;
            result.simulations=1;result.proof_plies=proof.plies;result.memory_bytes=tree.bytes();result.arena_nodes=tree.nodes.size();
            for(uint32_t i=0;i<root_count;++i){auto& edge=tree.edges[i];bool best=edge.move==proof.move;result.visits.emplace_back(edge.move,best?1:0);result.q_values.push_back(best?1.f:0.f);result.priors.push_back(edge.prior);}
            return result;
        }
    }
    if(limits.noise>0) {
        std::gamma_distribution<float> gamma(0.3f,1);std::vector<float> noise(root_count);float sum=0;
        for(float& n:noise){n=gamma(rng);sum+=n;}
        if(sum>0)for(uint32_t i=0;i<root_count;++i)tree.edges[i].prior=(1-limits.noise)*tree.edges[i].prior+limits.noise*noise[i]/sum;
    }
    std::vector<uint32_t> path_nodes,path_edges;path_nodes.reserve(128);path_edges.reserve(128);
    while(!cancelled()&&(limits.simulations<=0||result.simulations<limits.simulations)) {
        Board b=board;uint32_t current=0;
        path_nodes.clear();path_edges.clear();path_nodes.push_back(0);
        while(tree.nodes[current].expanded&&!tree.nodes[current].terminal) {
            const Node node=tree.nodes[current];double best_score=-1e100;
            uint32_t best=node.first;double exploration=std::sqrt(double(node.visits)+1);
            for(uint32_t i=node.first;i<node.first+node.count;++i) {
                const auto& e=tree.edges[i];
                double q=e.visits?e.sum/e.visits:(limits.fpu_reduction>0?node.value-limits.fpu_reduction:0);
                double score=q+(limits.cpuct*e.prior)*exploration/(1+e.visits);
                if(score>best_score){best_score=score;best=i;}
            }
            path_edges.push_back(best);b.make(tree.edges[best].move);
            current=tree.child(best);
            if(current==NONE)break;
            path_nodes.push_back(current);
            if(path_nodes.size()>=512)break;
        }
        float value;
        if(current==NONE||!tree.nodes[current].expanded)value=evaluate(tree,current,b,network,result,workspace);
        else value=tree.nodes[current].value;
        for(auto index:path_nodes)++tree.nodes[index].visits;
        for(auto it=path_edges.rbegin();it!=path_edges.rend();++it) {
            value=-value;
            auto& edge=tree.edges[*it];++edge.visits;edge.sum+=value;
        }
        result.max_depth=std::max(result.max_depth,int(path_edges.size()));++result.simulations;
    }
    uint32_t best=0;double total=0;
    for(uint32_t i=0;i<root_count;++i) {
        const auto& edge=tree.edges[i];total+=edge.sum;
        result.visits.emplace_back(edge.move,int(edge.visits));
        result.q_values.push_back(edge.visits?float(edge.sum/edge.visits):0);
        result.priors.push_back(edge.prior);
        const auto& previous=tree.edges[best];
        if(edge.visits>previous.visits||(edge.visits==previous.visits&&edge.prior>previous.prior))best=i;
    }
    if(limits.temperature>0&&result.simulations>0) {
        double maximum=-1e100;std::vector<double> weights;
        for(uint32_t i=0;i<root_count;++i)if(tree.edges[i].visits)maximum=std::max(maximum,std::log(double(tree.edges[i].visits))/limits.temperature);
        for(uint32_t i=0;i<root_count;++i)weights.push_back(tree.edges[i].visits?std::exp(std::log(double(tree.edges[i].visits))/limits.temperature-maximum):0);
        std::discrete_distribution<uint32_t> choose(weights.begin(),weights.end());best=choose(rng);
    }
    result.has_move=true;result.best=tree.edges[best].move;
    if(result.simulations)result.value=float(total/result.simulations);
    result.arena_nodes=tree.nodes.size();result.memory_bytes=tree.bytes();return result;
}
}
