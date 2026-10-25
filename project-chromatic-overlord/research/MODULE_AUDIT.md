# Actual factors and execution paths

This audit describes the checked-in implementation, not the capabilities implied
by project names. C/A/F/T are local OVERLORD designs inspired by the requested
SANABI/Laugh Tale ideas; no imported SANABI or Laugh Tale implementation was
found. No AEGIS implementation is present. `tactical_safety.hpp` is not called by
the search. All new algorithm flags and ResearchMetrics default to false.

| Factor | UCI option / condition | Actual operation | Work / dependencies |
|---|---|---|---|
| C | CounterexampleVerify | For each shortlisted root move, enumerate every legal opponent reply and minimize its evaluation from the root side. C alone evaluates replies at depth zero. | Requires complete reply coverage and all shortlisted comparisons; truncated comparison keeps MCTS move. Reply generation and NN work. |
| A | AdaptiveBudget | Once per root, normalized policy entropy and WDL uncertainty produce importance in [0.5,1.4]; multiply root CPuct by importance. Time mode may shorten search. | One root calculation; no online learning or full-tree uncertainty model. Changes allocation AND exploration, so its experiment cannot separate these mechanisms. |
| F | ForcingSearch | Shared tactical admission; allow checking moves in selective alpha-beta quiescence. | Capture, promotion, check-evasion search exists in verifier even without F. F adds noncapture checks and admission priority for checking roots. Depth bounded by VerificationDepth. |
| T | TacticalVerify | Shared tactical admission; leaf value = 0.25 NN WDL + 0.7 tanh(material/600). | A heuristic mixture, not a general tactical proof. Same NN architecture/weights. Material scan at each static evaluation. |
| P | Threads=2, SharedTree=true | Main thread owns mutable shared tree, creates/reserves deterministic leaf batches, inference workers evaluate read-only network with private workspaces, main thread backs up results. | Two CPU threads, virtual reservations and batch boundaries. This resource intervention also changes selection schedule; no concurrent unsynchronized tree writes. |

## Order

1. Establish hard deadline; if C/F/T enabled, MCTS deadline is at 75% of
   requested time. Total-NN mode similarly reserves 25% of the cap. A alone has
   no verifier reserve.
2. Root terminal/rule tests, root legal NN evaluation and expansion, then
   optional legacy MateProof (budget zero in the factorial).
3. A computes importance. Its time fraction is clamped to [0.65,1]; deadline is
   the minimum of this and the existing deadline. Root CPuct changes also in
   fixed-work modes.
4. Serial MCTS or shared-tree batched inference; reclaim arena under its limits.
   An edge Q is from its parent side-to-move perspective; a neural leaf value is
   from the leaf side to move and backup negates once per edge. Selection uses
   edge Q directly. Stable legal order resolves exact
   selection ties (strict greater-than).
5. If C/F/T enabled and time remains, sort root candidates with current MCTS
   choice first. If F or T, scan all root moves for an exact mate in one and admit
   captures/promotions (and F checks) before quiet visit leaders. This scan and
   its mate proof are shared F OR T cost, not independent contributions.
6. Check up to four candidates with equal remaining node quotas. C wraps reply
   enumeration around quiescence; F/T permit depth-2 descendants in that path.
   Without C, negate child quiescence. Check evasion has no stand pat; legal
   terminal rules are handled before heuristic values. Only complete comparison
   reranks; a 1e-5 margin preserves earlier candidate on near ties.
7. Record final selected move, retained-tree PV, timings; destroy transient arena
   and return JSON. PV does not represent a verifier proof line and may be short
   for a reranked root with zero MCTS visits.

All 16 Boolean combinations are valid, each supplemented by P=2 threads for 32
cells. `SharedTree=false` is an older root ensemble reference requiring the
verification options off; it is excluded. Other controls (CPuct, FPU, sparse head,
backend, depth/node caps) are fixed, not treated as free factors. MateProofNodes
is held at zero and its historical default256 is a separate control.

## Measurements and interpretation

ResearchMetrics is optional instrumentation; ON/OFF fixed-work tests and old
binary parity are recorded in `reports/factorial/preflight`. Timer overhead is
measured separately. Each result distinguishes attempted simulations, completed
leaf visits, cumulative node creations, NN/legal edge-list expansions, current
retained nodes, NN evaluations, verifier calls, admission work, evictions and
worker discarded evaluations. A legal edge-list expansion is not one new child
node per legal move. Root expansion is included in expansion count; root itself
is not a new child creation. The 513-bin histogram covers completed MCTS leaf
depths, with the final bin a cap for any deeper depth.

Timer fields are hierarchical: NN is within MCTS or verification; quiescence is
within verification; material/NN/check eligibility are inside quiescence. Do not
sum overlapping fields. Parallel NN time is the sum of worker evaluation times
and can overlap. Counterexample exclusive time excludes its quiescence calls.
Admission time is shared F/T, F time counts eligibility checks, T time counts
material scoring. Neither is the full causal marginal cost of enabling its
option; paired total CPU/wall differences provide that measurement.

Root NN has its own timer. `algorithm_ns` ends before PV extraction and arena
cleanup; `search_ns` includes return/arena cleanup, excludes JSON serialization;
driver wall includes JSON/protocol overhead. CPU delta is OS process accounting
(coarse on Windows); aggregate it over puzzles. Peak RSS is sampled after each
search and is not an instantaneous process peak. Arena peak fields are separate.

Fixed visits do not fix verifier work; a total NN *cap* controls another work
dimension and can stop early or leave work unused. Both report actual work,
candidate completion, and CPU cost. No fixed mode equalizes all instructions.
