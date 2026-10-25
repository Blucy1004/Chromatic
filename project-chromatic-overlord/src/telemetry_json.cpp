#include "zero/telemetry_json.hpp"
#include <algorithm>
#include <iomanip>
#include <numeric>
#include <ostream>
#include <sstream>
namespace zero {
namespace {
template<class T> void histogram(std::ostream& out,const T& a) {
    out<<'[';bool first=true;for(auto v:a){if(!first)out<<',';first=false;out<<v;}out<<']';
}
void summary(std::ostream& out,const DepthSummary& d) {
    out<<"{\"samples\":"<<d.samples<<",\"sum_depth\":"<<d.sum_depth<<",\"max_depth\":"<<d.max_depth
       <<",\"mean_depth\":"<<d.mean_depth<<",\"median_depth\":"<<d.median_depth<<'}';
}
void counters(std::ostream& out,const TelemetryCounters& c) {
    out<<"{\"node_creations\":"<<c.node_creations<<",\"legal_expansions\":"<<c.legal_expansions
       <<",\"root_legal_expansions\":"<<c.root_legal_expansions<<",\"child_fail_node_limit\":"<<c.child_fail_node_limit
       <<",\"child_fail_byte_capacity\":"<<c.child_fail_byte_capacity<<",\"expansion_fail_byte_capacity\":"<<c.expansion_fail_byte_capacity
       <<",\"neural_evaluations\":"<<c.neural_evaluations<<",\"non_retained_neural_evaluations\":"<<c.non_retained_neural_evaluations
       <<",\"resident_unexpanded_repeat\":"<<c.resident_unexpanded_repeat<<",\"terminal_evaluations\":"<<c.terminal_evaluations
       <<",\"terminal_leaf_endpoints\":"<<c.terminal_leaf_endpoints<<",\"depth_limit_endpoints\":"<<c.depth_limit_endpoints
       <<",\"proof_root_updates\":"<<c.proof_root_updates
       <<",\"attempted_simulations\":"<<c.attempted_simulations<<",\"aborted_simulations\":"<<c.aborted_simulations
       <<",\"prune_count\":"<<c.prune_count<<",\"nodes_evicted\":"<<c.nodes_evicted<<",\"edges_evicted\":"<<c.edges_evicted
       <<",\"discarded_neural_evaluations\":"<<c.discarded_neural_evaluations<<'}';
}
template<class T>void arena(std::ostream& out,const T& a) {
    out<<",\"nodes_size\":"<<a.nodes_size<<",\"nodes_capacity\":"<<a.nodes_capacity
       <<",\"edges_size\":"<<a.edges_size<<",\"edges_capacity\":"<<a.edges_capacity<<",\"memory_bytes\":"<<a.memory_bytes;
}
void event(std::ostream& out,const TelemetryEvent& e) {
    out<<"{\"occurred\":"<<(e.occurred?"true":"false")<<",\"reason\":\""<<e.reason<<"\",\"simulation\":"<<e.simulation
       <<",\"attempt\":"<<e.attempt<<",\"completed_simulations\":"<<e.completed_simulations
       <<",\"elapsed_ms\":"<<e.elapsed_ms<<",\"depth\":"<<e.depth;arena(out,e);out<<'}';
}
void root(std::ostream& out,const TelemetryRootSnapshot& r) {
    out<<"{\"uci\":\""<<r.move.uci()<<"\",\"prior\":"<<r.prior<<",\"q\":"<<r.q<<",\"visits\":"<<r.visits;
}
}
void write_telemetry_json(std::ostream& out,const SearchTelemetry& d) {
    out<<std::setprecision(12)<<"{\"schema_version\":2,\"depth\":";summary(out,d.depth);
    out<<",\"leaf_depth\":";histogram(out,d.leaf_depth);
    out<<",\"traversal_depth\":";histogram(out,d.traversal_depth);
    out<<",\"pre_failure_leaf_depth\":";histogram(out,d.pre_failure_leaf_depth);
    out<<",\"post_failure_leaf_depth\":";histogram(out,d.post_failure_leaf_depth);
    out<<",\"first_reserved_cap\":";event(out,d.first_reserved_cap);
    out<<",\"first_allocation_failure\":";event(out,d.first_allocation_failure);
    out<<",\"counters\":";counters(out,d.counters);
    out<<",\"root_initializations\":"<<d.root_initializations<<",\"proof_only\":"<<(d.proof_only?"true":"false")
       <<",\"telemetry_memory_bytes\":"<<d.telemetry_memory_bytes<<",\"roots\":[";
    for(size_t i=0;i<d.roots.size();++i){if(i)out<<',';root(out,d.roots[i]);out<<",\"leaf_depth\":";histogram(out,d.roots[i].leaf_depth);out<<'}';}
    out<<"],\"snapshots\":[";
    for(size_t i=0;i<d.snapshots.size();++i){if(i)out<<',';const auto& s=d.snapshots[i];
        out<<"{\"reason\":\""<<s.reason<<"\",\"simulation\":"<<s.simulation<<",\"elapsed_ms\":"<<s.elapsed_ms<<",\"depth\":";summary(out,s.depth);
        out<<",\"window_depth\":";summary(out,s.window_depth);
        out<<",\"leaf_depth\":";histogram(out,s.leaf_depth);
        out<<",\"window_leaf_depth\":";histogram(out,s.window_leaf_depth);arena(out,s);
        out<<",\"counters\":";counters(out,s.counters);out<<",\"roots\":[";
        for(size_t j=0;j<s.roots.size();++j){if(j)out<<',';root(out,s.roots[j]);out<<'}';}out<<"]}";
    }out<<"],\"prunes\":[";
    for(size_t i=0;i<d.prunes.size();++i){if(i)out<<',';const auto& p=d.prunes[i];
        out<<"{\"attempt\":"<<p.attempt<<",\"completed_simulations\":"<<p.completed_simulations<<",\"threshold\":"<<p.threshold
           <<",\"elapsed_ms\":"<<p.elapsed_ms<<",\"duration_ms\":"<<p.duration_ms<<",\"failed_depth\":"<<p.failed_depth
           <<",\"nodes_before\":"<<p.nodes_before<<",\"nodes_after\":"<<p.nodes_after<<",\"edges_before\":"<<p.edges_before<<",\"edges_after\":"<<p.edges_after
           <<",\"nodes_capacity\":"<<p.nodes_capacity<<",\"edges_capacity\":"<<p.edges_capacity<<",\"memory_bytes\":"<<p.memory_bytes
           <<",\"scratch_bytes\":"<<p.scratch_bytes<<",\"success\":"<<(p.success?"true":"false")
           <<",\"root_statistics_preserved\":"<<(p.root_statistics_preserved?"true":"false")
           <<",\"protected_root_children\":"<<(p.protected_root_children?"true":"false")<<",\"roots_before\":[";
        for(size_t j=0;j<p.roots_before.size();++j){if(j)out<<',';root(out,p.roots_before[j]);out<<'}';}out<<"],\"roots_after\":[";
        for(size_t j=0;j<p.roots_after.size();++j){if(j)out<<',';root(out,p.roots_after[j]);out<<'}';}out<<"]}";
    }out<<"]}";
}
std::string search_result_json(const SearchResult& r) {
    std::ostringstream out;out<<std::setprecision(12)<<"{\"bestmove\":\""<<(r.has_move?r.best.uci():"0000")<<"\",\"terminal\":"<<(r.terminal?"true":"false")
       <<",\"search_ns\":"<<r.search_ns<<",\"simulations\":"<<r.simulations<<",\"value\":"<<r.value<<",\"neural_evals\":"<<r.neural_evals
       <<",\"memory_bytes\":"<<r.memory_bytes<<",\"arena_nodes\":"<<r.arena_nodes<<",\"max_depth\":"<<r.max_depth
       <<",\"peak_memory_bytes\":"<<r.peak_memory_bytes<<",\"peak_auxiliary_bytes\":"<<r.peak_auxiliary_bytes
       <<",\"attempted_simulations\":"<<r.attempted_simulations<<",\"aborted_simulations\":"<<r.aborted_simulations
       <<",\"prune_count\":"<<r.prune_count<<",\"nodes_evicted\":"<<r.nodes_evicted<<",\"edges_evicted\":"<<r.edges_evicted
       <<",\"discarded_neural_evaluations\":"<<r.discarded_neural_evaluations<<",\"memory_exhausted\":"<<(r.memory_exhausted?"true":"false")
       <<",\"verification_nodes\":"<<r.verification_nodes<<",\"verification_evals\":"<<r.verification_evals
       <<",\"verified_candidates\":"<<r.verified_candidates<<",\"verification_changed\":"<<(r.verification_changed?"true":"false")
       <<",\"parallel_batches\":"<<r.parallel_batches<<",\"parallel_leaves\":"<<r.parallel_leaves<<",\"peak_inflight\":"<<r.peak_inflight
       <<",\"importance\":"<<r.importance
       <<",\"proof_nodes\":"<<r.proof_nodes<<",\"mate_proven\":"<<(r.mate_proven?"true":"false")<<",\"proof_plies\":"<<r.proof_plies<<",\"visits\":[";
    auto entry=[&](size_t i){out<<"{\"uci\":\""<<r.visits[i].first.uci()<<"\",\"action\":"<<action(r.visits[i].first)
        <<",\"n\":"<<r.visits[i].second<<",\"q\":"<<r.q_values[i]<<",\"prior\":"<<r.priors[i]<<'}';};
    for(size_t i=0;i<r.visits.size();++i){if(i)out<<',';entry(i);}out<<"],\"selected\":";
    auto selected=std::find_if(r.visits.begin(),r.visits.end(),[&](const auto& p){return r.has_move&&p.first==r.best;});
    if(selected!=r.visits.end())entry(size_t(selected-r.visits.begin()));else out<<"null";
    std::vector<size_t> order(r.visits.size());std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](size_t a,size_t b){return r.visits[a].second!=r.visits[b].second?r.visits[a].second>r.visits[b].second:r.priors[a]>r.priors[b];});
    out<<",\"top5\":[";for(size_t i=0;i<std::min<size_t>(5,order.size());++i){if(i)out<<',';entry(order[i]);}out<<']';
    out<<",\"research\":";
    if(!r.research)out<<"null";else {
        const auto& m=*r.research;
        out<<"{\"leaf_depth\":";histogram(out,m.leaf_depth);
        uint64_t count=0,sum=0;for(size_t i=0;i<m.leaf_depth.size();++i){count+=m.leaf_depth[i];sum+=m.leaf_depth[i]*i;}
        uint64_t cumulative=0;int low=0,high=0;
        for(size_t i=0;i<m.leaf_depth.size();++i){auto before=cumulative;cumulative+=m.leaf_depth[i];if(count&&before<=(count-1)/2&&cumulative>(count-1)/2)low=int(i);if(count&&before<=count/2&&cumulative>count/2)high=int(i);}
        out<<",\"depth_samples\":"<<count<<",\"mean_depth\":"<<(count?double(sum)/count:0)<<",\"median_depth\":"<<(low+high)/2.;
        auto field=[&](const char* key,uint64_t value){out<<",\""<<key<<"\":"<<value;};
        field("node_creations",m.node_creations);
        field("edge_expansions",m.edge_expansions);
        field("stored_edges",m.stored_edges);
        field("adaptive_calls",m.adaptive_calls);
        field("verification_calls",m.verification_calls);
        field("admission_calls",m.admission_calls);
        field("admission_nodes",m.admission_nodes);
        field("counterexample_calls",m.counterexample_calls);
        field("counterexample_replies",m.counterexample_replies);
        field("quiescence_nodes",m.quiescence_nodes);
        field("forcing_checks",m.forcing_checks);
        field("material_calls",m.material_calls);
        field("root_ns",m.root_ns);
        field("mcts_ns",m.mcts_ns);
        field("nn_mcts_ns",m.nn_mcts_ns);
        field("adaptive_ns",m.adaptive_ns);
        field("verification_ns",m.verification_ns);
        field("admission_ns",m.admission_ns);
        field("counterexample_exclusive_ns",m.counterexample_exclusive_ns);
        field("quiescence_ns",m.quiescence_ns);
        field("nn_verification_ns",m.nn_verification_ns);
        field("forcing_checks_ns",m.forcing_checks_ns);
        field("material_ns",m.material_ns);
        field("proof_ns",m.proof_ns);field("prune_ns",m.prune_ns);
        field("algorithm_ns",m.algorithm_ns);
        field("call_ns",m.call_ns);
        field("parallel_init_ns",m.parallel_init_ns);field("parallel_wait_handoff_ns",m.parallel_wait_handoff_ns);field("parallel_join_ns",m.parallel_join_ns);field("root_nn_ns",m.root_nn_ns);
        field("selection_ties",m.selection_ties);out<<",\"minimum_positive_gap\":"<<(m.minimum_positive_gap<1e90?m.minimum_positive_gap:0);
        out<<",\"pv\":[";for(size_t i=0;i<m.pv.size();++i){if(i)out<<',';out<<'"'<<m.pv[i].uci()<<'"';}out<<"]}";
    }
    out<<",\"telemetry_enabled\":"<<(r.telemetry?"true":"false");
    if(r.telemetry){out<<",\"telemetry\":";write_telemetry_json(out,*r.telemetry);}out<<'}';return out.str();
}
}
