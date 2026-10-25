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
       <<",\"simulations\":"<<r.simulations<<",\"value\":"<<r.value<<",\"neural_evals\":"<<r.neural_evals
       <<",\"root_neural_value\":"<<(r.root_neural_value?"true":"false")<<",\"root_raw_wdl\":["<<r.root_raw_wdl[0]<<","<<r.root_raw_wdl[1]<<","<<r.root_raw_wdl[2]<<"]"
       <<",\"root_raw_value\":"<<r.root_raw_value<<",\"root_search_value\":"<<r.root_search_value
       <<",\"memory_bytes\":"<<r.memory_bytes<<",\"arena_nodes\":"<<r.arena_nodes<<",\"max_depth\":"<<r.max_depth
       <<",\"peak_memory_bytes\":"<<r.peak_memory_bytes<<",\"peak_auxiliary_bytes\":"<<r.peak_auxiliary_bytes
       <<",\"attempted_simulations\":"<<r.attempted_simulations<<",\"aborted_simulations\":"<<r.aborted_simulations
       <<",\"prune_count\":"<<r.prune_count<<",\"nodes_evicted\":"<<r.nodes_evicted<<",\"edges_evicted\":"<<r.edges_evicted
       <<",\"discarded_neural_evaluations\":"<<r.discarded_neural_evaluations<<",\"memory_exhausted\":"<<(r.memory_exhausted?"true":"false")
       <<",\"proof_nodes\":"<<r.proof_nodes<<",\"mate_proven\":"<<(r.mate_proven?"true":"false")<<",\"proof_plies\":"<<r.proof_plies<<",\"visits\":[";
    auto entry=[&](size_t i){out<<"{\"uci\":\""<<r.visits[i].first.uci()<<"\",\"action\":"<<action(r.visits[i].first)
        <<",\"n\":"<<r.visits[i].second<<",\"q\":"<<r.q_values[i]<<",\"prior\":"<<r.priors[i]<<'}';};
    for(size_t i=0;i<r.visits.size();++i){if(i)out<<',';entry(i);}out<<"],\"selected\":";
    auto selected=std::find_if(r.visits.begin(),r.visits.end(),[&](const auto& p){return r.has_move&&p.first==r.best;});
    if(selected!=r.visits.end())entry(size_t(selected-r.visits.begin()));else out<<"null";
    std::vector<size_t> order(r.visits.size());std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](size_t a,size_t b){return r.visits[a].second!=r.visits[b].second?r.visits[a].second>r.visits[b].second:r.priors[a]>r.priors[b];});
    out<<",\"top5\":[";for(size_t i=0;i<std::min<size_t>(5,order.size());++i){if(i)out<<',';entry(order[i]);}out<<']';
    out<<",\"telemetry_enabled\":"<<(r.telemetry?"true":"false");
    if(r.telemetry){out<<",\"telemetry\":";write_telemetry_json(out,*r.telemetry);}out<<'}';return out.str();
}
}
