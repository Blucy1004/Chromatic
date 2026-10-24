# Source and workflow audit

## Baseline and scope

Audited the original engine, depth instrumentation and memory-recycling variants,
including board rules, PUCT backup, WDL/policy inference, proof search, UCI and
telemetry. Compared the original trainer, ARC trainer, normalization/streaming,
checkpoint/legacy import, validation and arena contracts. The new source derives
from project-chromatic-memory. Preserved paths and hashes are recorded in
reports/preserved-sha256.json. No SANABI/Laugh Tale source was found by a
case-insensitive repository search. The new designs make no external equivalence
claim. The audit focuses on the active engine/training paths and their preserved
variants; historical raw reports and duplicated archived packages are evidence,
not separate implementations revalidated line by line.

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
- The initial shortlist verification missed low-prior queen captures. The second
  iteration admits captures/promotions/checks and scans root terminal mates under
  the same budget. Finite-horizon verification still cannot explore every root
  alternative beyond its shortlist.
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

## Validation environment diagnosis

The existing ARC process-tree I/O test initially failed under /tmp. A controlled
child-process experiment showed write_count/write_chars increase there while
physical write_bytes stays zero; the same code under /workspace reports physical
bytes. Running the unchanged suite with pytest --basetemp on /workspace/scratch
passes that assertion. Both original failure and corrected run are retained.
The six actual kill/resume tests were run from an identical isolated source copy
so their hard-coded report paths cannot overwrite historical tracked reports.
The copied legacy Python integration fixture used copyfile for a test engine;
on Linux that strips executable permissions. Only the candidate test was changed
to copy2. All integration/model parity, replay/learning and safety tests then pass.

Neural trainer batching validates masks, WDL normalization, finite labels and
confidence weights. Validation/test families and source prefix hashes are kept
separate; durable model/optimizer/RNG/replay checkpoints are confirmed with the
SQLite ledger before shard deletion. The new engine does not train, rewrite or
promote any checkpoint. Historical champion training data cannot be reconstructed
from this repository; no stronger data-leakage claim is made.
