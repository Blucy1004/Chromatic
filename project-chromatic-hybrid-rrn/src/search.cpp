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
DepthSummary summarize(const DepthHistogram& histogram) {
    DepthSummary result;
    for(size_t d=0;d<histogram.size();++d)if(histogram[d]) {
        result.samples+=histogram[d];result.sum_depth+=histogram[d]*d;result.max_depth=int(d);
    }
    if(!result.samples)return result;
    result.mean_depth=double(result.sum_depth)/double(result.samples);
    uint64_t low=(result.samples-1)/2,high=result.samples/2,cumulative=0;
    int low_depth=-1,high_depth=-1;
    for(size_t d=0;d<histogram.size();++d) {
        cumulative+=histogram[d];
        if(low_depth<0&&cumulative>low)low_depth=int(d);
        if(cumulative>high){high_depth=int(d);break;}
    }
    result.median_depth=(low_depth+high_depth)/2.;return result;
}
// Integer indices remain valid when vectors grow. No pointers outlive a reserve.
template<bool Measured>
class Arena {
public:
    std::vector<Node> nodes;
    std::vector<Edge> edges;
    const SearchLimits& limits;
    SearchResult& result;
    SearchTelemetry* telemetry;
    std::chrono::steady_clock::time_point start;
    uint64_t simulation=0,attempt=0;
    int depth=0;
    std::vector<uint8_t> evaluated;
    explicit Arena(const SearchLimits& l,SearchResult& r,SearchTelemetry* t,std::chrono::steady_clock::time_point s):limits(l),result(r),telemetry(t),start(s) {
        if(l.max_nodes<1||l.max_tree_bytes<65536)throw std::runtime_error("invalid tree memory budget");
        nodes.reserve(std::min<size_t>(256,l.max_nodes));nodes.emplace_back();
        edges.reserve(std::min<size_t>(1024,(l.max_tree_bytes-nodes.capacity()*sizeof(Node))/sizeof(Edge)));
        peak();
        if constexpr(Measured)reserved_cap();
    }
    size_t bytes() const{return nodes.capacity()*sizeof(Node)+edges.capacity()*sizeof(Edge);}
    void peak(){result.peak_memory_bytes=std::max(result.peak_memory_bytes,bytes());}
    void sync_counters() {
        if constexpr(Measured) {
            auto& c=telemetry->counters;
            c.attempted_simulations=result.attempted_simulations;c.aborted_simulations=result.aborted_simulations;
            c.prune_count=result.prune_count;c.nodes_evicted=result.nodes_evicted;c.edges_evicted=result.edges_evicted;
            c.discarded_neural_evaluations=result.discarded_neural_evaluations;
        }
    }
    void event(TelemetryEvent& e,const char* reason) {
        if constexpr(Measured)if(!e.occurred) {
            e.occurred=true;e.reason=reason;e.simulation=simulation;e.depth=depth;
            e.attempt=attempt;e.completed_simulations=uint64_t(result.simulations);
            e.elapsed_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            e.nodes_size=nodes.size();e.nodes_capacity=nodes.capacity();e.edges_size=edges.size();e.edges_capacity=edges.capacity();e.memory_bytes=bytes();
        }
    }
    void reserved_cap() {
        if constexpr(Measured)if(bytes()==limits.max_tree_bytes)event(telemetry->first_reserved_cap,"reserved_bytes_equal_budget");
    }
    uint32_t child(uint32_t edge) {
        if(edges[edge].child!=NONE)return edges[edge].child;
        if(nodes.size()>=limits.max_nodes) {
            if constexpr(Measured){++telemetry->counters.child_fail_node_limit;event(telemetry->first_allocation_failure,"child_node_limit");}
            return NONE;
        }
        if(nodes.size()==nodes.capacity()) {
            size_t available=(limits.max_tree_bytes-edges.capacity()*sizeof(Edge))/sizeof(Node);
            size_t capacity=std::min({limits.max_nodes,available,std::max<size_t>(64,nodes.capacity()*2)});
            if(capacity<=nodes.size()) {
                if constexpr(Measured){++telemetry->counters.child_fail_byte_capacity;event(telemetry->first_allocation_failure,"child_byte_capacity");}
                return NONE;
            }
            nodes.reserve(capacity);
            peak();
            if constexpr(Measured)reserved_cap();
        }
        auto index=uint32_t(nodes.size());nodes.emplace_back();edges[edge].child=index;
        if constexpr(Measured)++telemetry->counters.node_creations;
        return index;
    }
    // Check storage before invoking the NN: failed attempts must never back up
    // an unretained shallow value. The append below cannot fail its byte budget.
    bool reserve_children(size_t count) {
        size_t needed=edges.size()+count;
        if(needed>edges.capacity()) {
            size_t available=(limits.max_tree_bytes-nodes.capacity()*sizeof(Node))/sizeof(Edge);
            size_t capacity=std::min(available,std::max(needed,edges.capacity()*2));
            if(capacity<needed) {
                if constexpr(Measured){++telemetry->counters.expansion_fail_byte_capacity;event(telemetry->first_allocation_failure,"edge_byte_capacity");}
                return false;
            }
            edges.reserve(capacity);
            peak();
            if constexpr(Measured)reserved_cap();
        }
        return true;
    }
    void children(uint32_t index,const std::vector<Move>& moves,const Evaluation& eval) {
        float maximum=-std::numeric_limits<float>::infinity(),sum=0;
        for(auto m:moves)maximum=std::max(maximum,eval.policy[action(m)]);
        uint32_t first=uint32_t(edges.size());
        for(auto m:moves){float p=std::exp(eval.policy[action(m)]-maximum);sum+=p;edges.push_back({m,p,0,0,NONE});}
        for(size_t i=first;i<edges.size();++i)edges[i].prior/=sum;
        nodes[index].first=first;nodes[index].count=uint32_t(moves.size());nodes[index].expanded=true;
        if constexpr(Measured){++telemetry->counters.legal_expansions;if(index==0)++telemetry->counters.root_legal_expansions;}
    }
    // Reclaim low-visit descendants in the existing reservations. No second
    // tree is allocated: only keep flags, an index map and block order scratch.
    // All root edges and the statistics on every retained edge are preserved.
    bool prune() {
        const auto prune_start=std::chrono::steady_clock::now();
        TelemetryPrune record;
        if constexpr(Measured) {
            record.attempt=attempt;record.completed_simulations=uint64_t(result.simulations);
            record.failed_depth=depth;record.nodes_before=nodes.size();record.edges_before=edges.size();
            record.nodes_capacity=nodes.capacity();record.edges_capacity=edges.capacity();record.memory_bytes=bytes();
            record.elapsed_ms=std::chrono::duration<double,std::milli>(prune_start-start).count();
            for(uint32_t i=0;i<nodes[0].count;++i) {
                const auto& e=edges[nodes[0].first+i];record.roots_before.push_back({e.move,e.prior,e.visits?float(e.sum/e.visits):0,e.visits});
            }
        }
        const size_t old_nodes=nodes.size(),old_edges=edges.size();
        if constexpr(Measured)evaluated.resize(old_nodes,0);
        const size_t target_nodes=std::max<size_t>(1,nodes.capacity()/2);
        const size_t target_edges=std::max<size_t>(nodes[0].count,edges.capacity()/2);
        uint64_t threshold=2;
        size_t kept_nodes=0,kept_edges=0,scratch_bytes=0;
        bool progress=false;
        {
            std::vector<uint8_t> keep(old_nodes,0);
            std::vector<uint32_t> remap(old_nodes,NONE),blocks;
            // Protect existing root children when the minimum protected set can
            // still meet both half-capacity targets. Tiny node caps may relax it.
            size_t protected_nodes=1,protected_edges=nodes[0].count;
            for(uint32_t i=0;i<nodes[0].count;++i) {
                auto child=edges[nodes[0].first+i].child;
                if(child!=NONE){++protected_nodes;protected_edges+=nodes[child].count;}
            }
            const bool protect_root_children=protected_nodes<=target_nodes&&protected_edges<=target_edges;
            if constexpr(Measured)record.protected_root_children=protect_root_children;
            while(true) {
                std::fill(keep.begin(),keep.end(),uint8_t(0));keep[0]=1;
                kept_nodes=kept_edges=0;
                // Parent indices precede their children, including after every
                // compaction. This pass only marks; it never assigns new IDs.
                for(size_t old=0;old<old_nodes;++old)if(keep[old]) {
                    ++kept_nodes;kept_edges+=nodes[old].count;
                    const auto& node=nodes[old];
                    for(uint32_t i=node.first;i<node.first+node.count;++i) {
                        uint32_t child=edges[i].child;
                        if(child!=NONE&&((old==0&&protect_root_children)||uint64_t(nodes[child].visits)>=threshold))keep[child]=1;
                    }
                }
                if((kept_nodes<=target_nodes&&kept_edges<=target_edges)||threshold>uint64_t(std::numeric_limits<uint32_t>::max()))break;
                threshold*=2;
            }
            progress=kept_nodes<old_nodes||kept_edges<old_edges;
            if(progress) {
                // Ascending old IDs guarantee remap[old] <= old, so forward
                // node copies cannot overwrite nodes still waiting to be read.
                uint32_t next=0;
                blocks.reserve(kept_nodes);
                for(size_t old=0;old<old_nodes;++old)if(keep[old]) {
                    remap[old]=next++;
                    if(nodes[old].count)blocks.push_back(uint32_t(old));
                }
                std::sort(blocks.begin(),blocks.end(),[&](uint32_t a,uint32_t b){return nodes[a].first<nodes[b].first;});
                size_t write_edge=0;
                for(uint32_t old:blocks) {
                    const uint32_t first=nodes[old].first,count=nodes[old].count;
                    nodes[old].first=uint32_t(write_edge);
                    for(uint32_t j=0;j<count;++j) {
                        Edge edge=edges[first+j];
                        if(edge.child!=NONE)edge.child=remap[edge.child];
                        edges[write_edge++]=edge;
                    }
                }
                for(size_t old=0;old<old_nodes;++old)if(keep[old]) {
                    const auto dest=remap[old];nodes[dest]=nodes[old];
                    if constexpr(Measured)evaluated[dest]=evaluated[old];
                }
                nodes.resize(kept_nodes);edges.resize(kept_edges);
                if constexpr(Measured)if(evaluated.size()>kept_nodes)evaluated.resize(kept_nodes);
                ++result.prune_count;result.nodes_evicted+=old_nodes-kept_nodes;result.edges_evicted+=old_edges-kept_edges;
            }
            scratch_bytes=keep.capacity()*sizeof(uint8_t)+remap.capacity()*sizeof(uint32_t)+blocks.capacity()*sizeof(uint32_t);
            // This excludes diagnostics flags/result/serializer memory; it is
            // exactly the algorithm's auxiliary compaction scratch peak.
            result.peak_auxiliary_bytes=std::max(result.peak_auxiliary_bytes,scratch_bytes);
        }
        sync_counters();
        if constexpr(Measured) {
            record.threshold=threshold;record.nodes_after=nodes.size();record.edges_after=edges.size();
            record.scratch_bytes=scratch_bytes;record.success=progress;
            for(uint32_t i=0;i<nodes[0].count;++i) {
                const auto& e=edges[nodes[0].first+i];record.roots_after.push_back({e.move,e.prior,e.visits?float(e.sum/e.visits):0,e.visits});
            }
            record.root_statistics_preserved=record.roots_before.size()==record.roots_after.size();
            for(size_t i=0;i<record.roots_before.size()&&record.root_statistics_preserved;++i) {
                const auto& a=record.roots_before[i];const auto& b=record.roots_after[i];
                record.root_statistics_preserved=a.move==b.move&&a.prior==b.prior&&a.q==b.q&&a.visits==b.visits;
            }
            record.duration_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-prune_start).count();
            telemetry->prunes.push_back(std::move(record));
        }
        return progress;
    }
};
template<bool Measured>
float evaluate(Arena<Measured>& tree,uint32_t index,const Board& b,const Network& net,SearchResult& result,Network::Workspace& workspace,bool* terminal=nullptr,bool* success=nullptr) {
    if(success)*success=true;
    auto moves=b.legal();float value;
    if(b.terminal(moves,value)) {
        if constexpr(Measured){++tree.telemetry->counters.terminal_evaluations;if(terminal)*terminal=true;}
        if(index!=NONE){auto& n=tree.nodes[index];n.expanded=n.terminal=true;n.value=value;}
        return value;
    }
    if(index==NONE)throw std::logic_error("unretained leaf evaluation is forbidden");
    if(!tree.reserve_children(moves.size())){if(success)*success=false;return 0;}
    if constexpr(Measured) {
        ++tree.telemetry->counters.neural_evaluations;
        if(index==NONE)++tree.telemetry->counters.non_retained_neural_evaluations;
        else {
            if(tree.evaluated.size()<=index)tree.evaluated.resize(size_t(index)+1,0);
            if(tree.evaluated[index])++tree.telemetry->counters.resident_unexpanded_repeat;
            tree.evaluated[index]=1;
        }
    }
    ++result.neural_evals;auto eval=net.evaluate_legal(b,moves,workspace);
    const float raw_value=eval.wdl[0]-eval.wdl[2];
    value=tree.limits.value_zero?0.0f:raw_value;
    if(index==0) {
        // A uniform fallback is an evaluation result, not a neural value.
        result.root_neural_value=net.loaded();
        result.root_raw_wdl=eval.wdl;
        result.root_raw_value=raw_value;
        result.root_search_value=value;
    }
    if(index!=NONE){tree.nodes[index].value=value;tree.children(index,moves,eval);}
    return value;
}
template<bool Measured>
void snapshot(Arena<Measured>& tree,SearchTelemetry& t,const char* reason,uint64_t simulation,DepthHistogram& window) {
    if constexpr(Measured) {
        TelemetrySnapshot s;s.reason=reason;s.simulation=simulation;
        s.elapsed_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tree.start).count();
        s.leaf_depth=t.leaf_depth;s.window_leaf_depth=window;s.depth=summarize(s.leaf_depth);s.window_depth=summarize(window);
        s.nodes_size=tree.nodes.size();s.nodes_capacity=tree.nodes.capacity();s.edges_size=tree.edges.size();s.edges_capacity=tree.edges.capacity();s.memory_bytes=tree.bytes();s.counters=t.counters;
        for(uint32_t i=0;i<tree.nodes[0].count;++i) {
            const auto& edge=tree.edges[tree.nodes[0].first+i];
            s.roots.push_back({edge.move,edge.prior,edge.visits?float(edge.sum/edge.visits):0,edge.visits});
        }
        t.snapshots.push_back(std::move(s));window.fill(0);
    }
}
template<bool Measured>
void finish_telemetry(Arena<Measured>& tree,SearchResult& result,DepthHistogram& window) {
    if constexpr(Measured) {
        tree.sync_counters();
        auto& t=*result.telemetry;t.depth=summarize(t.leaf_depth);
        uint64_t tail=0;
        for(size_t i=t.leaf_depth.size();i-- >0;){tail+=t.leaf_depth[i];t.traversal_depth[i]=tail;}
        // Proof updates retain their historical result.visits semantics, but are
        // never represented as fabricated MCTS leaf endpoints.
        for(size_t i=0;i<t.roots.size()&&i<result.visits.size();++i) {
            t.roots[i].visits=uint64_t(result.visits[i].second);t.roots[i].q=result.q_values[i];t.roots[i].prior=result.priors[i];
        }
        snapshot(tree,t,"final",t.depth.samples,window);
        t.telemetry_memory_bytes=sizeof(SearchTelemetry)+t.roots.capacity()*sizeof(TelemetryRoot)+t.snapshots.capacity()*sizeof(TelemetrySnapshot);
        for(const auto& s:t.snapshots)t.telemetry_memory_bytes+=s.roots.capacity()*sizeof(TelemetryRootSnapshot)+s.reason.capacity();
        t.telemetry_memory_bytes+=t.prunes.capacity()*sizeof(TelemetryPrune);
        for(const auto& p:t.prunes)t.telemetry_memory_bytes+=(p.roots_before.capacity()+p.roots_after.capacity())*sizeof(TelemetryRootSnapshot);
    }
}
template<bool Measured>
SearchResult search_impl(const Board& board,const Network& network,const SearchLimits& limits,const std::atomic<bool>& stop) {
    const auto start=std::chrono::steady_clock::now();
    auto cancelled=[&](){return stop.load(std::memory_order_relaxed)||(limits.milliseconds>0&&std::chrono::steady_clock::now()-start>=std::chrono::milliseconds(limits.milliseconds));};
    SearchResult result;
    if constexpr(Measured)result.telemetry=std::make_shared<SearchTelemetry>();
    Arena<Measured> tree(limits,result,result.telemetry.get(),start);Network::Workspace workspace;
    DepthHistogram window{};
    bool root_stored=true;
    result.value=evaluate(tree,0,board,network,result,workspace,nullptr,&root_stored);result.terminal=tree.nodes[0].terminal;
    result.arena_nodes=tree.nodes.size();result.memory_bytes=tree.bytes();
    if(result.terminal){finish_telemetry(tree,result,window);return result;}
    if(!root_stored||!tree.nodes[0].expanded)throw std::runtime_error("tree budget cannot hold root");
    std::mt19937_64 rng(limits.seed);
    const uint32_t root_count=tree.nodes[0].count;
    if constexpr(Measured) {
        result.telemetry->roots.resize(root_count);
        for(uint32_t i=0;i<root_count;++i){result.telemetry->roots[i].move=tree.edges[i].move;result.telemetry->roots[i].prior=tree.edges[i].prior;}
        if(result.telemetry->first_reserved_cap.occurred)snapshot(tree,*result.telemetry,"root_initialization+first_reserved_cap",0,window);
    }
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
            if constexpr(Measured){result.telemetry->proof_only=true;result.telemetry->counters.proof_root_updates=1;}
            finish_telemetry(tree,result,window);
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
        ++result.attempted_simulations;
        tree.simulation=uint64_t(result.simulations)+1;tree.attempt=result.attempted_simulations;
        if constexpr(Measured){tree.depth=0;tree.sync_counters();}
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
            if constexpr(Measured)tree.depth=int(path_edges.size());
            current=tree.child(best);
            if(current==NONE)break;
            path_nodes.push_back(current);
            if(path_nodes.size()>=512)break;
        }
        float value=0;bool terminal_endpoint=false,stored=current!=NONE;
        if(current!=NONE&&!tree.nodes[current].expanded)value=evaluate(tree,current,b,network,result,workspace,&terminal_endpoint,&stored);
        else if(current!=NONE) {
            value=tree.nodes[current].value;
            if constexpr(Measured){terminal_endpoint=tree.nodes[current].terminal;if(!terminal_endpoint&&path_nodes.size()>=512)++result.telemetry->counters.depth_limit_endpoints;}
        }
        if(!stored) {
            // The failed path contributes neither a neural leaf value nor a
            // virtual visit. All saved indices become invalid after pruning.
            ++result.aborted_simulations;tree.sync_counters();
            if(cancelled())break;
            if(!tree.prune()){result.memory_exhausted=true;break;}
            if constexpr(Measured)snapshot(tree,*result.telemetry,"recycle",uint64_t(result.simulations),window);
            continue;
        }
        for(auto index:path_nodes)++tree.nodes[index].visits;
        for(auto it=path_edges.rbegin();it!=path_edges.rend();++it) {
            value=-value;
            auto& edge=tree.edges[*it];++edge.visits;edge.sum+=value;
        }
        result.max_depth=std::max(result.max_depth,int(path_edges.size()));++result.simulations;
        if constexpr(Measured) {
            auto& t=*result.telemetry;const size_t d=path_edges.size();
            ++t.leaf_depth[d];++window[d];
            if(!path_edges.empty())++t.roots[path_edges.front()-tree.nodes[0].first].leaf_depth[d];
            if(t.first_allocation_failure.occurred)++t.post_failure_leaf_depth[d];else ++t.pre_failure_leaf_depth[d];
            if(terminal_endpoint)++t.counters.terminal_leaf_endpoints;
            const uint64_t n=uint64_t(result.simulations);
            bool first_reserved=t.first_reserved_cap.occurred&&t.first_reserved_cap.simulation==n;
            bool first_failure=t.first_allocation_failure.occurred&&t.first_allocation_failure.simulation==n;
            bool early=n>=64&&n<=4096&&(n&(n-1))==0;
            bool interval=limits.telemetry_interval>0&&n%uint64_t(limits.telemetry_interval)==0;
            if(first_reserved||first_failure||early||interval) {
                std::string reason;
                auto add=[&](const char* text){if(!reason.empty())reason+='+';reason+=text;};
                if(first_reserved)add("first_reserved_cap");if(first_failure)add("first_allocation_failure");if(early)add("early");if(interval)add("interval");
                snapshot(tree,t,reason.c_str(),n,window);
            }
        }
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
    result.arena_nodes=tree.nodes.size();result.memory_bytes=tree.bytes();finish_telemetry(tree,result,window);return result;
}
}
SearchResult search(const Board& board,const Network& network,const SearchLimits& limits,const std::atomic<bool>& stop) {
    return limits.telemetry?search_impl<true>(board,network,limits,stop):search_impl<false>(board,network,limits,stop);
}
}
