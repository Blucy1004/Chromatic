#include "zero/search.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>

using namespace zero;

namespace {
void require(bool ok,const std::string& message) {
    if(!ok)throw std::runtime_error(message);
}
template<class T>uint64_t count(const T& histogram) {
    return std::accumulate(histogram.begin(),histogram.end(),uint64_t(0));
}
template<class T>uint64_t weighted_count(const T& histogram) {
    uint64_t total=0;
    for(size_t d=0;d<histogram.size();++d)total+=uint64_t(d)*histogram[d];
    return total;
}
template<class T>double exact_median(const T& histogram) {
    const uint64_t total=count(histogram);
    if(!total)return 0;
    const uint64_t lower=(total-1)/2,upper=total/2;
    uint64_t before=0;
    int left=-1,right=-1;
    for(size_t d=0;d<histogram.size();++d) {
        if(left<0&&before+histogram[d]>lower)left=int(d);
        if(before+histogram[d]>upper){right=int(d);break;}
        before+=histogram[d];
    }
    return (left+right)/2.0;
}
void same_result(const SearchResult& off,const SearchResult& on) {
    require(!off.telemetry,"off search must not allocate telemetry");
    require(bool(on.telemetry),"on search must return telemetry");
    require(off.has_move==on.has_move&&off.best==on.best,"telemetry changed selected move");
    require(off.terminal==on.terminal&&off.simulations==on.simulations,"telemetry changed search endpoint count");
    require(off.neural_evals==on.neural_evals&&off.value==on.value,"telemetry changed NN/backup result");
    require(off.visits==on.visits&&off.q_values==on.q_values&&off.priors==on.priors,"telemetry changed root statistics");
    require(off.max_depth==on.max_depth&&off.arena_nodes==on.arena_nodes&&off.memory_bytes==on.memory_bytes,"telemetry changed tree storage/depth");
    require(off.proof_nodes==on.proof_nodes&&off.proof_plies==on.proof_plies&&off.mate_proven==on.mate_proven,"telemetry changed proof result");
    require(off.attempted_simulations==on.attempted_simulations&&off.aborted_simulations==on.aborted_simulations,"telemetry changed failed-attempt handling");
    require(off.prune_count==on.prune_count&&off.nodes_evicted==on.nodes_evicted&&off.edges_evicted==on.edges_evicted,"telemetry changed compaction decisions");
    require(off.discarded_neural_evaluations==on.discarded_neural_evaluations&&off.memory_exhausted==on.memory_exhausted,"telemetry changed bounded-search termination");
    require(off.peak_memory_bytes==on.peak_memory_bytes,"telemetry changed peak arena reservation");
    require(off.peak_auxiliary_bytes==on.peak_auxiliary_bytes,"telemetry changed compaction scratch reservation");
}
void validate(const SearchResult& result) {
    require(bool(result.telemetry),"missing telemetry");
    const auto& t=*result.telemetry;
    const uint64_t samples=count(t.leaf_depth);
    const uint64_t expected=t.proof_only?0:uint64_t(result.simulations);
    require(samples==expected,"one leaf endpoint per MCTS simulation");
    require(t.depth.samples==samples,"summary sample count");
    require(t.depth.sum_depth==weighted_count(t.leaf_depth),"summary depth sum");
    require(t.depth.median_depth==exact_median(t.leaf_depth),"exact histogram median");
    const double mean=samples?double(weighted_count(t.leaf_depth))/samples:0;
    require(std::abs(t.depth.mean_depth-mean)<1e-12,"histogram mean");
    int deepest=0;
    for(size_t d=0;d<t.leaf_depth.size();++d)if(t.leaf_depth[d])deepest=int(d);
    require(t.depth.max_depth==deepest,"histogram max depth");
    require(result.max_depth==deepest,"legacy max matches MCTS endpoint max");
    require(t.leaf_depth[0]==0,"root initialization is not a sampled leaf endpoint");
    require(t.traversal_depth[0]==samples,"root traversal accounting");
    uint64_t tail=0;
    for(size_t d=t.leaf_depth.size();d-->1;) {
        tail+=t.leaf_depth[d];
        require(t.traversal_depth[d]==tail,"edge traversal depth equals leaf endpoint tail sum");
    }
    for(size_t d=0;d<t.leaf_depth.size();++d) {
        require(t.pre_failure_leaf_depth[d]+t.post_failure_leaf_depth[d]==t.leaf_depth[d],"before/after histograms partition endpoint visits");
        uint64_t by_root=0;
        for(const auto& root:t.roots)by_root+=root.leaf_depth[d];
        require(by_root==t.leaf_depth[d],"root move histograms partition endpoint visits");
    }
    require(t.counters.node_creations+1>=t.counters.nodes_evicted,"evictions cannot exceed cumulative node creations");
    require(t.counters.node_creations+1-t.counters.nodes_evicted==result.arena_nodes,"cumulative creations minus evictions equals resident nodes");
    require(t.counters.neural_evaluations==uint64_t(result.neural_evals),"NN evaluations include root initialization");
    require(t.counters.legal_expansions>=t.counters.root_legal_expansions,"root expansion subset");
    require(t.counters.legal_expansions<=t.counters.node_creations+1,"at most one legal expansion per retained node");
    require(t.counters.non_retained_neural_evaluations==0,"allocation failures must not evaluate unretained leaves");
    require(t.counters.attempted_simulations==result.attempted_simulations,"attempt counter output");
    require(t.counters.aborted_simulations==result.aborted_simulations,"aborted counter output");
    require(t.counters.prune_count==result.prune_count,"compaction counter output");
    require(t.counters.nodes_evicted==result.nodes_evicted&&t.counters.edges_evicted==result.edges_evicted,"eviction counter output");
    if(!t.proof_only)require(result.attempted_simulations==uint64_t(result.simulations)+result.aborted_simulations,"aborted attempts never add completed visits");
    const uint64_t failures=t.counters.child_fail_node_limit+t.counters.child_fail_byte_capacity+t.counters.expansion_fail_byte_capacity;
    require(t.first_allocation_failure.occurred==(failures>0),"first failure event agrees with actual allocation failure counters");
    if(t.first_allocation_failure.occurred) {
        const auto first=t.first_allocation_failure.simulation;
        require(first>=1&&first<=samples+1,"failure event identifies the next completed simulation, even if it cannot complete");
        const auto pre=std::min(samples,first-1);
        require(count(t.pre_failure_leaf_depth)==pre,"pre-failure distribution excludes the first retried completion");
        require(count(t.post_failure_leaf_depth)==samples-pre,"failed attempts themselves are absent from endpoint histograms");
        require(t.first_allocation_failure.attempt>=1&&t.first_allocation_failure.attempt<=result.attempted_simulations,"failure event records an actual attempt");
    } else {
        require(t.pre_failure_leaf_depth==t.leaf_depth,"all endpoints precede failure when no failure occurs");
        require(count(t.post_failure_leaf_depth)==0,"no post-failure endpoints without failure");
    }
    require(t.roots.size()==result.visits.size(),"root move count");
    for(size_t i=0;i<t.roots.size();++i) {
        const auto& root=t.roots[i];
        require(root.move==result.visits[i].first,"root move order");
        require(root.prior==result.priors[i]&&root.q==result.q_values[i],"root prior/Q output");
        require(root.visits==uint64_t(result.visits[i].second),"root visit output");
        if(!t.proof_only)require(count(root.leaf_depth)==root.visits,"root endpoint histogram sum");
    }
    DepthHistogram previous{};
    uint64_t previous_samples=0;
    double previous_ms=0;
    require(!t.snapshots.empty(),"on telemetry includes a final snapshot");
    for(const auto& snapshot:t.snapshots) {
        const uint64_t snapshot_samples=count(snapshot.leaf_depth);
        require(snapshot.depth.samples==snapshot_samples,"snapshot cumulative sample count");
        require(snapshot.window_depth.samples==count(snapshot.window_leaf_depth),"snapshot window sample count");
        if(!t.proof_only)require(snapshot.simulation==snapshot_samples,"snapshot follows completed backup/accounting");
        require(snapshot_samples>=previous_samples,"snapshot sample counts are monotonic");
        require(snapshot.elapsed_ms>=previous_ms,"snapshot clocks are monotonic");
        require(snapshot.depth.sum_depth==weighted_count(snapshot.leaf_depth),"snapshot cumulative depth sum");
        require(snapshot.window_depth.sum_depth==weighted_count(snapshot.window_leaf_depth),"snapshot window depth sum");
        require(snapshot.depth.median_depth==exact_median(snapshot.leaf_depth),"snapshot cumulative median");
        require(snapshot.window_depth.median_depth==exact_median(snapshot.window_leaf_depth),"snapshot window median");
        for(size_t d=0;d<previous.size();++d) {
            require(previous[d]+snapshot.window_leaf_depth[d]==snapshot.leaf_depth[d],"snapshot windows exactly partition cumulative histograms");
        }
        uint64_t root_visits=0;
        for(const auto& root:snapshot.roots)root_visits+=root.visits;
        if(!t.proof_only)require(root_visits==snapshot_samples,"snapshot root visits match completed simulations");
        previous=snapshot.leaf_depth;
        previous_samples=snapshot_samples;
        previous_ms=snapshot.elapsed_ms;
    }
    require(previous==t.leaf_depth,"final snapshot includes all leaf endpoints");
}
SearchResult paired(const Board& board,const Network& net,SearchLimits limits) {
    std::atomic<bool> stop=false;
    limits.telemetry=false;
    const auto off=search(board,net,limits,stop);
    limits.telemetry=true;
    limits.telemetry_interval=31;
    const auto on=search(board,net,limits,stop);
    same_result(off,on);
    validate(on);
    return on;
}
}

int main(int argc,char** argv) {try {
    Network net;
    if(argc>1)net.load(argv[1]);
    const auto board=Board::from_fen(START_FEN);
    SearchLimits limits;
    limits.seed=2026;
    limits.simulations=1;
    auto one=paired(board,net,limits);
    require(one.telemetry->leaf_depth[1]==1,"one simulation ends at depth one");
    require(one.telemetry->counters.node_creations==1,"one retained child creation");
    require(one.telemetry->counters.root_legal_expansions==1,"exactly one root initialization expansion");
    require(one.telemetry->counters.legal_expansions==2,"root and child legal expansions");

    limits.simulations=8192;
    auto standard=paired(board,net,limits);
    require(standard.memory_bytes<=limits.max_tree_bytes,"default tree remains bounded");
    require(standard.telemetry->counters.child_fail_node_limit==0,"ample default node capacity");
    require(standard.telemetry->counters.child_fail_byte_capacity==0,"ample default byte capacity");

    limits.simulations=512;
    limits.max_tree_bytes=65536;
    auto byte_bound=paired(board,net,limits);
    require(byte_bound.memory_bytes<=limits.max_tree_bytes,"small byte budget remains bounded");
    require(byte_bound.telemetry->counters.child_fail_byte_capacity+byte_bound.telemetry->counters.expansion_fail_byte_capacity>0,"small tree triggers a real byte allocation failure");

    limits.max_tree_bytes=64*1024*1024;
    limits.max_nodes=16;
    limits.simulations=128;
    auto node_bound=paired(board,net,limits);
    require(node_bound.arena_nodes<=16,"node cap remains bounded after recycling");
    require(node_bound.telemetry->counters.node_creations>15,"recycling permits actual node creation beyond the resident cap");
    require(node_bound.nodes_evicted>0&&node_bound.prune_count>0,"node cap recycles storage instead of evaluating unretained leaves");
    require(node_bound.telemetry->counters.child_fail_node_limit>0,"node cap failure is reported separately");
    require(node_bound.telemetry->counters.child_fail_byte_capacity==0,"node cap is not mislabelled byte cap");

    limits.noise=.25f;
    limits.temperature=1;
    paired(board,net,limits);

    limits=SearchLimits{};
    limits.simulations=32;
    auto terminal=paired(Board::from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"),net,limits);
    require(terminal.terminal&&terminal.simulations==0,"terminal root has no MCTS simulation");
    require(terminal.telemetry->depth.samples==0,"terminal initialization not counted as leaf sample");
    require(terminal.telemetry->counters.node_creations==0,"terminal root creates no child");
    require(terminal.telemetry->counters.root_legal_expansions==0,"terminal root creates no legal edges");

    limits.proof_nodes=12000;
    limits.proof_plies=5;
    limits.simulations=1;
    auto proof=paired(Board::from_fen("7k/8/8/4K3/8/2Q5/8/8 w - - 0 1"),net,limits);
    require(proof.mate_proven&&proof.telemetry->proof_only,"proof-only result explicitly identified");
    require(proof.simulations==1&&proof.telemetry->depth.samples==0,"proof root update is not a fake MCTS leaf");

    std::cout<<"depth telemetry tests passed; model="<<(argc>1?"loaded":"uniform")<<'\n';
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
