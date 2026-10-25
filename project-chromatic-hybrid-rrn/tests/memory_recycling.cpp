#include "zero/search.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>

using namespace zero;
namespace {
void require(bool condition,const std::string& message) {
    if(!condition)throw std::runtime_error(message);
}
uint64_t sum(const DepthHistogram& histogram) {
    return std::accumulate(histogram.begin(),histogram.end(),uint64_t(0));
}
void equal_roots(const std::vector<TelemetryRootSnapshot>& before,
                 const std::vector<TelemetryRootSnapshot>& after) {
    require(before.size()==after.size(),"compaction changed root legal move count");
    for(size_t i=0;i<before.size();++i) {
        require(before[i].move==after[i].move,"compaction changed root move identity/order");
        require(before[i].prior==after[i].prior,"compaction changed root policy");
        require(before[i].visits==after[i].visits,"compaction discarded or fabricated root visits");
        require(before[i].q==after[i].q,"compaction changed root Q");
    }
}
void validate(const SearchResult& r,const SearchLimits& limits) {
    require(bool(r.telemetry),"recycling tests require telemetry");
    const auto& t=*r.telemetry;
    const auto& c=t.counters;
    require(r.memory_bytes<=limits.max_tree_bytes&&r.peak_memory_bytes<=limits.max_tree_bytes,"arena reservation exceeds byte budget");
    require(r.arena_nodes<=limits.max_nodes,"resident nodes exceed configured count");
    require(c.node_creations+1>=c.nodes_evicted,"invalid cumulative node eviction count");
    require(c.node_creations+1-c.nodes_evicted==r.arena_nodes,"creations/evictions do not reconcile with resident tree");
    require(c.non_retained_neural_evaluations==0,"unallocated child must never be NN evaluated");
    require(c.discarded_neural_evaluations==0,"edge capacity must be checked before NN evaluation");
    require(c.resident_unexpanded_repeat==0,"recycled node IDs must not inherit stale evaluation flags");
    require(c.neural_evaluations==uint64_t(r.neural_evals),"NN call accounting differs from search result");
    require(c.attempted_simulations==r.attempted_simulations,"attempt counter mismatch");
    require(c.aborted_simulations==r.aborted_simulations,"aborted counter mismatch");
    require(c.prune_count==r.prune_count,"prune counter mismatch");
    require(c.nodes_evicted==r.nodes_evicted&&c.edges_evicted==r.edges_evicted,"eviction output mismatch");
    require(r.attempted_simulations==uint64_t(r.simulations)+r.aborted_simulations,"failed allocation attempt added completed visits");
    require(sum(t.leaf_depth)==uint64_t(r.simulations),"failed attempts must not create leaf endpoints");
    uint64_t root_visits=0;
    for(auto [move,n]:r.visits){(void)move;root_visits+=uint64_t(n);}
    require(root_visits==uint64_t(r.simulations),"root visits do not equal successful backups");
    require(t.roots.size()==r.visits.size(),"root telemetry size");
    for(size_t i=0;i<t.roots.size();++i) {
        require(sum(t.roots[i].leaf_depth)==uint64_t(r.visits[i].second),"root endpoint histogram includes failed attempts");
    }
    require(sum(t.pre_failure_leaf_depth)+sum(t.post_failure_leaf_depth)==uint64_t(r.simulations),"phase partition lost or fabricated endpoints");
    uint64_t successful_prunes=0,evicted_nodes=0,evicted_edges=0;
    for(const auto& p:t.prunes) {
        require(p.memory_bytes<=limits.max_tree_bytes,"compaction increased reserved arena beyond budget");
        require(p.completed_simulations<=uint64_t(r.simulations),"compaction reports future completed backups");
        require(p.attempt>=1&&p.attempt<=r.attempted_simulations,"compaction attempt id is not real");
        equal_roots(p.roots_before,p.roots_after);
        require(p.root_statistics_preserved,"compaction root invariant flag");
        uint64_t before_visits=0;
        for(const auto& root:p.roots_before)before_visits+=root.visits;
        require(before_visits==p.completed_simulations,"failed attempt must not back up before compaction");
        require(p.nodes_after<=p.nodes_before&&p.edges_after<=p.edges_before,"compaction grew logical tree storage");
        if(p.success) {
            ++successful_prunes;
            require(p.threshold>=2,"reclamation must prune low-visit subtrees");
            require(p.nodes_after<p.nodes_before||p.edges_after<p.edges_before,"successful compaction reclaimed no slots");
            evicted_nodes+=p.nodes_before-p.nodes_after;
            evicted_edges+=p.edges_before-p.edges_after;
        }
    }
    require(successful_prunes==r.prune_count,"successful prune events disagree with counter");
    require(evicted_nodes==r.nodes_evicted&&evicted_edges==r.edges_evicted,"event reclamation differs from cumulative evictions");
    const uint64_t failures=c.child_fail_node_limit+c.child_fail_byte_capacity+c.expansion_fail_byte_capacity;
    require(failures==r.aborted_simulations,"one capacity failure must produce exactly one aborted attempt");
    if(t.first_allocation_failure.occurred) {
        const auto& e=t.first_allocation_failure;
        require(e.simulation==e.completed_simulations+1,"failure completed-ordinal definition");
        require(sum(t.pre_failure_leaf_depth)==std::min<uint64_t>(r.simulations,e.completed_simulations),"phase boundary not tied to successful backups");
    }
}
SearchResult paired(const Board& board,const Network& network,SearchLimits limits) {
    const auto fen=board.fen();const auto history=board.history;
    std::atomic<bool> stop=false;
    limits.telemetry=false;
    const auto off=search(board,network,limits,stop);
    limits.telemetry=true;limits.telemetry_interval=97;
    auto on=search(board,network,limits,stop);
    require(!off.telemetry&&bool(on.telemetry),"optional telemetry switch");
    require(off.best==on.best&&off.has_move==on.has_move,"telemetry changed move under reclamation");
    require(off.simulations==on.simulations&&off.neural_evals==on.neural_evals,"telemetry changed completed work");
    require(off.value==on.value&&off.visits==on.visits&&off.q_values==on.q_values&&off.priors==on.priors,"telemetry changed PUCT root statistics");
    require(off.arena_nodes==on.arena_nodes&&off.max_depth==on.max_depth&&off.memory_bytes==on.memory_bytes,"telemetry changed retained tree");
    require(off.attempted_simulations==on.attempted_simulations&&off.aborted_simulations==on.aborted_simulations,"telemetry changed aborted attempts");
    require(off.prune_count==on.prune_count&&off.nodes_evicted==on.nodes_evicted&&off.edges_evicted==on.edges_evicted,"telemetry changed reclamation");
    require(off.memory_exhausted==on.memory_exhausted,"telemetry changed fallback termination");
    require(off.peak_memory_bytes==on.peak_memory_bytes&&off.peak_auxiliary_bytes==on.peak_auxiliary_bytes,"telemetry changed algorithm memory reservation");
    require(board.fen()==fen&&board.history==history,"search or compaction mutated input Board/history");
    validate(on,limits);return on;
}
}

int main(int argc,char** argv){try {
    Network network;
    if(argc>1)network.load(argv[1]);
    const auto start=Board::from_fen(START_FEN);
    SearchLimits limits;limits.seed=2026;limits.proof_nodes=0;

    // Before capacity pressure the new search still uses one real node per
    // first nonterminal visit and never enters the reclamation path.
    limits.simulations=128;
    auto prefix=paired(start,network,limits);
    require(prefix.simulations==128&&prefix.prune_count==0&&prefix.aborted_simulations==0,"pre-cap search unexpectedly reclaimed storage");
    require(prefix.telemetry->counters.non_retained_neural_evaluations==0,"pre-cap NN storage guard");

    // The byte cap must permit continued real stored expansion after several
    // pressure events, without treating failed allocation as a simulation.
    limits.simulations=8192;limits.max_tree_bytes=1024*1024;limits.max_nodes=1000000;
    auto bytes=paired(start,network,limits);
    require(bytes.simulations==8192&&!bytes.memory_exhausted,"byte cap prevented requested completed backups despite reclaimable slots");
    require(bytes.prune_count>=2&&bytes.nodes_evicted>0&&bytes.edges_evicted>0,"small arena must recycle repeatedly");
    require(bytes.telemetry->counters.node_creations>bytes.arena_nodes,"successful searches must actually recreate retained nodes after pressure");
    require(sum(bytes.telemetry->post_failure_leaf_depth)>0,"no completed work after first pressure event");
    uint64_t deeper=0;for(size_t d=3;d<bytes.telemetry->post_failure_leaf_depth.size();++d)deeper+=bytes.telemetry->post_failure_leaf_depth[d];
    require(deeper>0,"all post-pressure work stopped at unretained depth-one/two leaves");

    // A count limit exercises the same recovery independently of byte capacity.
    limits.max_tree_bytes=64*1024*1024;limits.max_nodes=16;limits.simulations=512;
    auto nodes=paired(start,network,limits);
    require(nodes.simulations==512&&!nodes.memory_exhausted,"node slots should be reusable");
    require(nodes.telemetry->counters.child_fail_node_limit>0&&nodes.telemetry->counters.child_fail_byte_capacity==0,"node limit mislabeled as byte limit");
    require(nodes.prune_count>0&&nodes.nodes_evicted>0,"node cap repeated static leaf evaluation instead of recycling");

    // With only a root slot there is genuinely no possible progress. Returning
    // a legal zero-visit fallback is honest; fabricating requested N is not.
    limits.max_nodes=1;limits.simulations=1000;
    auto impossible=paired(start,network,limits);
    require(impossible.memory_exhausted&&impossible.simulations==0,"irreducible capacity limit must stop without invented visits");
    require(impossible.aborted_simulations==1&&impossible.attempted_simulations==1,"irreducible failure must not retry forever");
    require(impossible.neural_evals==1,"irreducible child failure must not reevaluate the NN beyond root initialization");
    require(impossible.has_move,"memory-exhausted root must retain a legal fallback move");
    const auto legal=start.legal();require(std::find(legal.begin(),legal.end(),impossible.best)!=legal.end(),"memory-exhausted fallback is illegal");

    // Reclamation must preserve the input repetition context and special-move
    // state. Board reconstruction on every retry always starts from this root.
    Board history=start;for(auto move:{"g1f3","g8f6","f3g1","f6g8"})history.play(move);
    require(history.repetitions()==2,"twofold fixture");
    limits.max_nodes=16;limits.simulations=256;
    paired(history,network,limits);
    paired(Board::from_fen("1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1"),network,limits);
    paired(Board::from_fen("4k3/8/8/r4pPK/8/8/8/8 w - f6 0 1"),network,limits);

    std::cout<<"memory recycling regression tests passed; model="<<(argc>1?"loaded":"uniform")<<'\n';
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
