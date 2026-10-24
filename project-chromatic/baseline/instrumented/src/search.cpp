#include "zero/search.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
namespace zero {
namespace {
struct Node;
struct Edge {Move move;float prior=0;int visits=0;double sum=0;std::unique_ptr<Node> child;};
struct Node {std::vector<Edge> edges;bool expanded=false,terminal=false;float value=0;int visits=0;};
float expand(Node& n,const Board& b,const Network& net) {
    n.expanded=true;auto moves=b.legal();
    if(b.terminal(moves,n.value)){n.terminal=true;return n.value;}
    auto eval=net.evaluate(b);n.value=eval.wdl[0]-eval.wdl[2];float max=-std::numeric_limits<float>::infinity(),sum=0;
    for(auto m:moves)max=std::max(max,eval.policy[action(m)]);
    for(auto m:moves){float p=std::exp(eval.policy[action(m)]-max);sum+=p;n.edges.push_back({m,p,0,0,nullptr});}
    for(auto& e:n.edges)e.prior/=sum;return n.value;
}
}
SearchResult search(const Board& board,const Network& network,const SearchLimits& limits,const std::atomic<bool>& stop) {
    SearchResult result;Node root;expand(root,board,network);result.value=root.value;result.terminal=root.terminal;
    if(root.terminal)return result;
    std::mt19937_64 rng(limits.seed);size_t node_count=1;
    if(limits.noise>0){std::gamma_distribution<float> gamma(0.3f,1);std::vector<float> noise(root.edges.size());float sum=0;for(float& n:noise){n=gamma(rng);sum+=n;}if(sum>0)for(size_t i=0;i<noise.size();++i)root.edges[i].prior=(1-limits.noise)*root.edges[i].prior+limits.noise*noise[i]/sum;}
    auto start=std::chrono::steady_clock::now();
    while(!stop.load(std::memory_order_relaxed) && (limits.simulations<=0||result.simulations<limits.simulations)) {
        if(limits.milliseconds>0 && std::chrono::steady_clock::now()-start>=std::chrono::milliseconds(limits.milliseconds))break;
        Board b=board;Node* n=&root;std::vector<Node*> nodes{n};std::vector<Edge*> edges;
        while(n->expanded&&!n->terminal) {
            Edge* best=nullptr;double score=-1e100;
            for(auto& e:n->edges) {
                double q=e.visits?e.sum/e.visits:0;
                double u=limits.cpuct*e.prior*std::sqrt(double(n->visits)+1)/(1+e.visits);
                if(q+u>score){score=q+u;best=&e;}
            }
            edges.push_back(best);b.make(best->move);
            if(!best->child){if(node_count>=limits.max_nodes)break;best->child=std::make_unique<Node>();++node_count;}
            n=best->child.get();nodes.push_back(n);
            if(nodes.size()>=512)break;
        }
        float value;
        if(edges.size()>=nodes.size()) { // Memory bound: evaluate leaf without retaining a child.
            auto moves=b.legal();if(!b.terminal(moves,value)){auto e=network.evaluate(b);value=e.wdl[0]-e.wdl[2];}
        } else if(!n->expanded)value=expand(*n,b,network);
        else value=n->value;
        for(auto* visited:nodes)++visited->visits;
        for(auto it=edges.rbegin();it!=edges.rend();++it){value=-value;++(*it)->visits;(*it)->sum+=value;}
        ++result.simulations;
    }
    Edge* best=&root.edges.front();double total=0;
    for(auto& e:root.edges){result.visits.emplace_back(e.move,e.visits);total+=e.sum;if(e.visits>best->visits||(e.visits==best->visits&&e.prior>best->prior))best=&e;}
    if(limits.temperature>0 && result.simulations>0) {
        double maxlog=-1e100;std::vector<double> weights;for(auto& e:root.edges)if(e.visits)maxlog=std::max(maxlog,std::log(double(e.visits))/limits.temperature);
        for(auto& e:root.edges)weights.push_back(e.visits?std::exp(std::log(double(e.visits))/limits.temperature-maxlog):0);
        std::discrete_distribution<size_t> choose(weights.begin(),weights.end());best=&root.edges[choose(rng)];
    }
    result.has_move=true;result.best=best->move;if(result.simulations)result.value=float(total/result.simulations);return result;
}
}
