#include "zero/search.hpp"
#include "zero/proof.hpp"
#include "zero/verification.hpp"
#include "zero/leaf_pool.hpp"
#include <unordered_map>
#include <unordered_set>
#include <optional>
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
        if(result.research)++result.research->node_creations;
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
        if(result.research){++result.research->edge_expansions;result.research->stored_edges+=moves.size();}
        nodes[index].first=first;nodes[index].count=uint32_t(moves.size());nodes[index].expanded=true;
        if constexpr(Measured){++telemetry->counters.legal_expansions;if(index==0)++telemetry->counters.root_legal_expansions;}
    }
    // Reclaim low-visit descendants in the existing reservations. No second
    // tree is allocated: only keep flags, an index map and block order scratch.
    // All root edges and the statistics on every retained edge are preserved.
    bool prune() {
        ResearchTimer timer(result.research?&result.research->prune_ns:nullptr);
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
    ++result.neural_evals;Evaluation eval;
    {ResearchTimer timer(result.research?&result.research->nn_mcts_ns:nullptr);eval=net.evaluate_legal(b,moves,workspace);}
    value=eval.wdl[0]-eval.wdl[2];
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
// Synchronous shared-tree waves: selection/backup/recycling have one owner;
// only immutable NN work crosses threads. Pending leaves cannot be selected
// twice, and virtual losses disappear at the wave boundary before compaction.
template<bool Measured, class Cancelled>
void parallel_simulations(Arena<Measured>& tree,const Board& board,const Network& network,
    const SearchLimits& limits,SearchResult& result,DepthHistogram& window,Cancelled cancelled) {
    // Threads includes the coordinator. It evaluates one leaf itself while
    // Threads-1 workers process the other reservations; no hidden extra CPU.
    const auto pool_start=result.research?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
    LeafPool pool(limits.threads-1,bool(result.research));
    if(result.research)result.research->parallel_init_ns=uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-pool_start).count());
    Network::Workspace coordinator_workspace;
    struct Pending {
        uint32_t node;std::vector<uint32_t> nodes,edges;
        std::vector<Move> moves;float value=0;bool terminal=false,neural=false;
        std::future<Evaluation> future;std::optional<Board> deferred;
    };
    while(!cancelled()&&(limits.simulations<=0||result.simulations<limits.simulations)) {
        std::vector<Pending> wave;wave.reserve(size_t(limits.threads));
        std::unordered_set<uint32_t> pending;
        std::unordered_map<uint32_t,unsigned> virtual_visits;
        bool reclaim=false;
        for(int worker=0;worker<limits.threads&&!cancelled()&&
            (limits.simulations<=0||result.simulations+int(wave.size())<limits.simulations);++worker) {
            Board b=board;Pending task{};task.node=0;task.nodes.push_back(0);
            bool blocked=false;
            while(tree.nodes[task.node].expanded&&!tree.nodes[task.node].terminal) {
                auto node=tree.nodes[task.node];double best_score=-1e100,runner_up=-1e100;uint32_t best=NONE;
                for(uint32_t i=node.first;i<node.first+node.count;++i) {
                    const auto& e=tree.edges[i];
                    if(e.child!=NONE&&pending.count(e.child))continue;
                    const auto found=virtual_visits.find(i);
                    const unsigned in_flight=found==virtual_visits.end()?0:found->second;
                    const double n=double(e.visits)+in_flight;
                    double q=n?(e.sum-double(in_flight))/n:(limits.fpu_reduction>0?node.value-limits.fpu_reduction:0);
                    double cpuct=limits.cpuct*(task.node==0&&limits.adaptive_budget?result.importance:1.f);
                    double score=q+cpuct*e.prior*std::sqrt(double(node.visits)+1)/(1+n);
                    if(score>best_score){if(result.research)runner_up=best_score;best_score=score;best=i;}
                    else if(result.research&&score>runner_up)runner_up=score;
                }
                if(result.research&&runner_up>-1e90){double gap=best_score-runner_up;if(gap==0)++result.research->selection_ties;else result.research->minimum_positive_gap=std::min(result.research->minimum_positive_gap,gap);}
                if(best==NONE){blocked=true;break;}
                task.edges.push_back(best);b.make(tree.edges[best].move);
                task.node=tree.child(best);
                if(task.node==NONE)break;
                task.nodes.push_back(task.node);if(task.nodes.size()>=512)break;
            }
            if(blocked)break;
            ++result.attempted_simulations;tree.attempt=result.attempted_simulations;
            tree.simulation=uint64_t(result.simulations)+wave.size()+1;tree.depth=int(task.edges.size());
            if(task.node==NONE){++result.aborted_simulations;reclaim=true;break;}
            if(!tree.nodes[task.node].expanded) {
                task.moves=b.legal();task.terminal=b.terminal(task.moves,task.value);
                if(task.terminal) {
                    auto& node=tree.nodes[task.node];node.expanded=node.terminal=true;node.value=task.value;
                    if constexpr(Measured)++tree.telemetry->counters.terminal_evaluations;
                } else {
                    if(!tree.reserve_children(task.moves.size())){++result.aborted_simulations;reclaim=true;break;}
                    // Reserve the complete legal edge block before NN inference.
                    // No failed-storage value can enter the tree or visits.
                    tree.children(task.node,task.moves,Evaluation{});
                    pending.insert(task.node);task.neural=true;
                    ++result.neural_evals;
                    if constexpr(Measured) {
                        ++tree.telemetry->counters.neural_evaluations;
                        if(tree.evaluated.size()<=task.node)tree.evaluated.resize(size_t(task.node)+1,0);
                        if(tree.evaluated[task.node])++tree.telemetry->counters.resident_unexpanded_repeat;
                        tree.evaluated[task.node]=1;
                    }
                    if(worker+1==limits.threads)task.deferred=std::move(b);
                    else task.future=pool.submit(network,std::move(b),task.moves);
                }
            } else {task.value=tree.nodes[task.node].value;task.terminal=tree.nodes[task.node].terminal;}
            for(auto edge:task.edges)++virtual_visits[edge];
            wave.push_back(std::move(task));
        }
        if(!wave.empty())++result.parallel_batches;
        result.peak_inflight=std::max(result.peak_inflight,int(pending.size()));
        // Run the coordinator's leaf before waiting for worker futures.
        for(auto& task:wave)if(task.deferred) {
            std::promise<Evaluation> ready;task.future=ready.get_future();
            try {Evaluation evaluated;{ResearchTimer timer(result.research?&result.research->nn_mcts_ns:nullptr);evaluated=network.evaluate_legal(*task.deferred,task.moves,coordinator_workspace);}ready.set_value(evaluated);}
            catch(...) {ready.set_exception(std::current_exception());}
        }
        for(auto& task:wave) {
            if(task.neural) {
                Evaluation eval;{ResearchTimer timer(result.research?&result.research->parallel_wait_handoff_ns:nullptr);eval=task.future.get();}task.value=eval.wdl[0]-eval.wdl[2];
                tree.nodes[task.node].value=task.value;
                const auto node=tree.nodes[task.node];float maximum=-std::numeric_limits<float>::infinity(),sum=0;
                for(auto m:task.moves)maximum=std::max(maximum,eval.policy[action(m)]);
                for(uint32_t i=node.first;i<node.first+node.count;++i){auto& e=tree.edges[i];e.prior=std::exp(eval.policy[action(e.move)]-maximum);sum+=e.prior;}
                for(uint32_t i=node.first;i<node.first+node.count;++i)tree.edges[i].prior/=sum;
                ++result.parallel_leaves;
            }
            for(auto index:task.nodes)++tree.nodes[index].visits;
            float value=task.value;
            for(auto it=task.edges.rbegin();it!=task.edges.rend();++it){value=-value;auto& e=tree.edges[*it];++e.visits;e.sum+=value;}
            ++result.simulations;result.max_depth=std::max(result.max_depth,int(task.edges.size()));
            if(result.research)++result.research->leaf_depth[task.edges.size()];
            if constexpr(Measured) {
                auto& t=*result.telemetry;size_t d=task.edges.size();++t.leaf_depth[d];++window[d];
                if(!task.edges.empty())++t.roots[task.edges.front()-tree.nodes[0].first].leaf_depth[d];
                if(t.first_allocation_failure.occurred)++t.post_failure_leaf_depth[d];else ++t.pre_failure_leaf_depth[d];
                if(task.terminal)++t.counters.terminal_leaf_endpoints;
                if(!task.terminal&&d>=511)++t.counters.depth_limit_endpoints;
                tree.sync_counters();
                if(limits.telemetry_interval>0&&result.simulations%limits.telemetry_interval==0)snapshot(tree,t,"interval",uint64_t(result.simulations),window);
            }
        }
        // All jobs are joined and every real backup committed before pruning.
        if(reclaim) {
            tree.sync_counters();if(cancelled())break;
            if(!tree.prune()){result.memory_exhausted=true;break;}
            if constexpr(Measured)snapshot(tree,*result.telemetry,"recycle",uint64_t(result.simulations),window);
        } else if(wave.empty())break;
    }
    {ResearchTimer timer(result.research?&result.research->parallel_join_ns:nullptr);pool.close();}
    if(result.research)result.research->nn_mcts_ns+=pool.neural_ns();
}
template<bool Measured>
void finish_research(Arena<Measured>& tree,SearchResult& result) {
    if(!result.research)return;
    auto& m=*result.research;
    m.algorithm_ns=uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-tree.start).count());
    if(!result.has_move)return;
    uint32_t current=0;
    for(int depth=0;depth<32;++depth) {
        const auto& node=tree.nodes[current];if(!node.expanded||node.terminal||!node.count)break;
        uint32_t best=node.first;
        if(depth==0){for(uint32_t i=node.first;i<node.first+node.count;++i)if(tree.edges[i].move==result.best){best=i;break;}}
        else for(uint32_t i=node.first;i<node.first+node.count;++i) {
            const auto& a=tree.edges[i];const auto& b=tree.edges[best];
            if(a.visits>b.visits||(a.visits==b.visits&&a.prior>b.prior))best=i;
        }
        m.pv.push_back(tree.edges[best].move);current=tree.edges[best].child;if(current==NONE)break;
    }
}

template<bool Measured>
SearchResult search_impl(const Board& board,const Network& network,const SearchLimits& limits,const std::atomic<bool>& stop) {
    const auto start=std::chrono::steady_clock::now();
    if(limits.threads<1||limits.threads>32||limits.verification_nodes<1||limits.verification_depth<1||limits.verification_depth>8||limits.verification_candidates<1)
        throw std::runtime_error("invalid Overlord search limits");
    const bool verify=limits.counterexample||limits.forcing_search||limits.tactical_verify;
    const auto hard_deadline=limits.milliseconds>0?start+std::chrono::milliseconds(limits.milliseconds):std::chrono::steady_clock::time_point::max();
    auto search_deadline=hard_deadline;
    if(verify&&limits.milliseconds>0)search_deadline=start+std::chrono::milliseconds(std::max<int64_t>(1,int64_t(limits.milliseconds)*3/4));
    SearchResult result;
    if(limits.research_metrics)result.research=std::make_shared<ResearchMetrics>();
    int mcts_neural_cap=limits.neural_limit;
    if(verify&&mcts_neural_cap>0)mcts_neural_cap=std::max(1,mcts_neural_cap*3/4);
    auto cancelled=[&](){return stop.load(std::memory_order_relaxed)||std::chrono::steady_clock::now()>=search_deadline||(mcts_neural_cap>0&&result.neural_evals>=mcts_neural_cap);};
    if constexpr(Measured)result.telemetry=std::make_shared<SearchTelemetry>();
    Arena<Measured> tree(limits,result,result.telemetry.get(),start);Network::Workspace workspace;
    DepthHistogram window{};
    bool root_stored=true;
    {ResearchTimer timer(result.research?&result.research->root_ns:nullptr);result.value=evaluate(tree,0,board,network,result,workspace,nullptr,&root_stored);}
    if(result.research)result.research->root_nn_ns=result.research->nn_mcts_ns;
    result.terminal=tree.nodes[0].terminal;
    result.arena_nodes=tree.nodes.size();result.memory_bytes=tree.bytes();
    if(result.terminal){finish_telemetry(tree,result,window);finish_research(tree,result);return result;}
    if(!root_stored||!tree.nodes[0].expanded)throw std::runtime_error("tree budget cannot hold root");
    std::mt19937_64 rng(limits.seed);
    const uint32_t root_count=tree.nodes[0].count;
    if constexpr(Measured) {
        result.telemetry->roots.resize(root_count);
        for(uint32_t i=0;i<root_count;++i){result.telemetry->roots[i].move=tree.edges[i].move;result.telemetry->roots[i].prior=tree.edges[i].prior;}
        if(result.telemetry->first_reserved_cap.occurred)snapshot(tree,*result.telemetry,"root_initialization+first_reserved_cap",0,window);
    }
    if(limits.proof_nodes>0&&!cancelled()) {
        ResearchTimer timer(result.research?&result.research->proof_ns:nullptr);
        std::vector<Move> moves;for(uint32_t i=0;i<root_count;++i)moves.push_back(tree.edges[i].move);
        int microseconds=limits.milliseconds>0?int(std::max<int64_t>(100, std::min<int64_t>(10000,int64_t(limits.milliseconds)*1000/4))):0;
        auto proof=verify_mate(board,moves,limits.proof_plies,limits.proof_nodes,stop,microseconds);
        result.proof_nodes=proof.nodes;
        if(proof.proven) {
            // One actual root update backed by a proof; do not invent requested visits.
            result.best=proof.move;result.has_move=result.mate_proven=true;result.value=1;
            result.simulations=1;result.proof_plies=proof.plies;result.memory_bytes=tree.bytes();result.arena_nodes=tree.nodes.size();
            for(uint32_t i=0;i<root_count;++i){auto& edge=tree.edges[i];bool best=edge.move==proof.move;result.visits.emplace_back(edge.move,best?1:0);result.q_values.push_back(best?1.f:0.f);result.priors.push_back(edge.prior);}
            if constexpr(Measured){result.telemetry->proof_only=true;result.telemetry->counters.proof_root_updates=1;}
            finish_telemetry(tree,result,window);finish_research(tree,result);
            return result;
        }
    }
    if(limits.noise>0) {
        std::gamma_distribution<float> gamma(0.3f,1);std::vector<float> noise(root_count);float sum=0;
        for(float& n:noise){n=gamma(rng);sum+=n;}
        if(sum>0)for(uint32_t i=0;i<root_count;++i)tree.edges[i].prior=(1-limits.noise)*tree.edges[i].prior+limits.noise*noise[i]/sum;
    }
    if(limits.adaptive_budget) {
        ResearchTimer timer(result.research?&result.research->adaptive_ns:nullptr);
        if(result.research)++result.research->adaptive_calls;
        std::vector<float> priors;for(uint32_t i=0;i<root_count;++i)priors.push_back(tree.edges[i].prior);
        result.importance=budget_importance(priors,result.value);
        if(limits.milliseconds>0) {
            // movetime is always a hard ceiling; simple/confident positions
            // release some of their budget. Equal-node tests keep their exact N.
            auto fraction=std::clamp(result.importance,.65f,1.f);
            auto adaptive=start+std::chrono::milliseconds(std::max(1,int(limits.milliseconds*fraction)));
            search_deadline=std::min(search_deadline,adaptive);
        }
    }
    const auto mcts_start=result.research?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
    if(limits.threads>1)parallel_simulations(tree,board,network,limits,result,window,cancelled);
    std::vector<uint32_t> path_nodes,path_edges;path_nodes.reserve(128);path_edges.reserve(128);
    while(limits.threads==1&&!cancelled()&&(limits.simulations<=0||result.simulations<limits.simulations)) {
        ++result.attempted_simulations;
        tree.simulation=uint64_t(result.simulations)+1;tree.attempt=result.attempted_simulations;
        if constexpr(Measured){tree.depth=0;tree.sync_counters();}
        Board b=board;uint32_t current=0;
        path_nodes.clear();path_edges.clear();path_nodes.push_back(0);
        while(tree.nodes[current].expanded&&!tree.nodes[current].terminal) {
            const Node node=tree.nodes[current];double best_score=-1e100,runner_up=-1e100;
            uint32_t best=node.first;double exploration=std::sqrt(double(node.visits)+1);
            for(uint32_t i=node.first;i<node.first+node.count;++i) {
                const auto& e=tree.edges[i];
                double q=e.visits?e.sum/e.visits:(limits.fpu_reduction>0?node.value-limits.fpu_reduction:0);
                double score=q+(limits.cpuct*(current==0&&limits.adaptive_budget?result.importance:1.f)*e.prior)*exploration/(1+e.visits);
                if(score>best_score){if(result.research)runner_up=best_score;best_score=score;best=i;}
                else if(result.research&&score>runner_up)runner_up=score;
            }
            if(result.research&&runner_up>-1e90){double gap=best_score-runner_up;if(gap==0)++result.research->selection_ties;else result.research->minimum_positive_gap=std::min(result.research->minimum_positive_gap,gap);}
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
        if(result.research)++result.research->leaf_depth[path_edges.size()];
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
                if(first_reserved)add("first_reserved_cap");
                if(first_failure)add("first_allocation_failure");
                if(early)add("early");
                if(interval)add("interval");
                snapshot(tree,t,reason.c_str(),n,window);
            }
        }
    }
    if(result.research)result.research->mcts_ns+=uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-mcts_start).count());
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
    result.arena_nodes=tree.nodes.size();result.memory_bytes=tree.bytes();finish_telemetry(tree,result,window);
    if(verify&&!stop.load(std::memory_order_relaxed)&&std::chrono::steady_clock::now()<hard_deadline)
        verify_candidates(board,network,limits,stop,hard_deadline,result);
    finish_research(tree,result);
    return result;
}
}
SearchResult search(const Board& board,const Network& network,const SearchLimits& limits,const std::atomic<bool>& stop) {
    if(limits.threads>1&&!limits.shared_tree) {
        if(limits.threads>32||limits.max_tree_bytes<size_t(limits.threads)*65536||limits.max_nodes<size_t(limits.threads)||limits.telemetry||limits.counterexample||limits.adaptive_budget||limits.forcing_search||limits.tactical_verify)
            throw std::runtime_error("root reference needs >=64KiB per worker, node capacity, telemetry and verification modules off");
        std::vector<std::future<SearchResult>> jobs;
        const int count=limits.simulations>0?std::min(limits.threads,limits.simulations):limits.threads;
        for(int i=0;i<count;++i) {
            auto part=limits;part.threads=1;part.max_tree_bytes/=size_t(count);part.max_nodes/=size_t(count);
            if(limits.simulations>0)part.simulations=limits.simulations/count+(i<limits.simulations%count?1:0);
            jobs.push_back(std::async(std::launch::async,[&,part]{return search_impl<false>(board,network,part,stop);}));
        }
        SearchResult combined;bool first=true;
        for(auto& job:jobs) {
            auto r=job.get();if(first){combined=r;first=false;continue;}
            if(r.mate_proven&&!combined.mate_proven){combined=r;continue;}
            if(combined.mate_proven)continue;
            for(size_t i=0;i<r.visits.size();++i) {
                const auto a=combined.visits[i].second,b=r.visits[i].second;
                combined.q_values[i]=a+b?(combined.q_values[i]*a+r.q_values[i]*b)/float(a+b):0;
                combined.visits[i].second+=b;
            }
            combined.simulations+=r.simulations;combined.neural_evals+=r.neural_evals;
            combined.attempted_simulations+=r.attempted_simulations;combined.aborted_simulations+=r.aborted_simulations;
            combined.memory_bytes+=r.memory_bytes;combined.peak_memory_bytes+=r.peak_memory_bytes;
            combined.peak_auxiliary_bytes+=r.peak_auxiliary_bytes;combined.arena_nodes+=r.arena_nodes;
            combined.prune_count+=r.prune_count;combined.nodes_evicted+=r.nodes_evicted;combined.edges_evicted+=r.edges_evicted;
            combined.max_depth=std::max(combined.max_depth,r.max_depth);
        }
        if(combined.has_move&&!combined.mate_proven) {
            size_t best=0;double total=0;
            for(size_t i=0;i<combined.visits.size();++i) {
                total+=combined.q_values[i]*combined.visits[i].second;
                if(combined.visits[i].second>combined.visits[best].second||
                    (combined.visits[i].second==combined.visits[best].second&&combined.priors[i]>combined.priors[best]))best=i;
            }
            combined.best=combined.visits[best].first;combined.value=combined.simulations?float(total/combined.simulations):0;
        }
        return combined;
    }
    return limits.telemetry?search_impl<true>(board,network,limits,stop):search_impl<false>(board,network,limits,stop);
}
}
