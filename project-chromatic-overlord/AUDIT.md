# Source and workflow audit

## Baseline and scope

Audited the original engine, depth instrumentation and memory-recycling variants,
including board rules, PUCT backup, WDL/policy inference, proof search, UCI and
telemetry. Compared the original trainer, ARC trainer, normalization/streaming,
checkpoint/legacy import, validation and arena contracts. The new source derives
from project-chromatic-memory. Preserved paths and hashes are recorded in
reports/preserved-sha256.json. No SANABI/Laugh Tale source was found by a
case-insensitive repository search. The new designs make no external equivalence
claim.

## Findings driving implementation

- Edge Q is from the parent side to move. Leaf WDL is from the leaf side; every
  reversed edge backup negates exactly once. FPU uses the parent node value.
  Synthetic child-win weights test this in serial and parallel paths.
- The earlier arena could repeatedly back up unretained leaves; the memory
  variant already fixes this by checking storage before inference and preserving
  root statistics during compaction. Overlord reuses that fix and tests it under
  parallel pressure. No transposition table is added: history-dependent draw
  states are never merged using a board-only key.
- Network weights are immutable during search. SIMD/sparse policy selects the
  same FP32 rows; each worker needs its own vectors. Profiling state is already
  thread_local. No BLAS runtime or hidden inference worker pool is introduced.
- UCI joins an active search before changing weights, positions or options.
  A search thread owns a board copy. New leaf workers are all joined before
  returning bestmove. Extra JSON info reports verification separately from N.
- A low-policy winning root move may remain unseen with a small simulation
  budget and MateProof disabled. A first test assuming 256 loaded-model playouts
  must find mate-in-one failed; the raw result is retained. The sign test now
  isolates the value convention with a uniform policy and synthetic weights;
  the loaded-model exact-mate test separately exercises existing MateProof.
- The new shortlist verification cannot discover moves outside its shortlist.
  Counterexample verification searches all replies at the first opponent node,
  but finite-horizon WDL is not an exhaustive strategic refutation. Quiescence
  can miss quiet defenses, sacrifices and deeper forced wins; stand pat is
  disallowed in check, but a depth-limited frontier still uses a heuristic.
- AdaptiveBudget is a bounded local entropy/uncertainty heuristic, not calibrated
  win-probability prediction. It may save time or harm strength; it stays off.
- Parallel waves trade fewer races and deterministic commit order for scheduling
  overhead, stale within-wave decisions and no asynchronous backup. Repeated
  root choice experiments are necessary; nodes/second alone does not measure Elo.
- Parallel matching needs a defined control: original UCI has no Threads option.
  Two independent legacy-algorithm trees with pooled visits are provided as an
  equal-worker reference; one-worker features also compare directly against the
  unchanged historical memory-recycling executable.

## Rules and limits

Board history is copied through all verifier branches and worker tasks. Existing
legal generation handles promotion, castling and pinned en passant. Terminal
checks give checkmate precedence over the 50-move rule, then stalemate,
insufficient material and repetition. The existing convention treats threefold
and 50 moves as immediately terminal rather than distinguishing optional claims
from automatic FIDE 5-fold/75-move draws. Match validation uses python-chess
claim_draw=True consistently. No new FIDE claim protocol is introduced.

Original trainer/ARC tests remain unchanged. Local fixture smoke training already
completed during onboarding. Regression outcomes distinguish opt-in recovery
suites, hardware/Windows skips and the existing container I/O-counter assertion.
No output from historical reports is relabeled as a new validation run.
