# CHROMATIC 1.1 — bounded tree memory recycling

This is a separate search-only revision. The original release in
`outputs/project-chromatic` and the prior depth-instrumented version in
`outputs/project-chromatic-depth` are preserved. The model is byte-identical to
their `default.zero`. No training, move filters or opening book were added.

## What changed

The previous arena continued evaluating and backing up a selected leaf when its
child could not be stored. A failed legal-edge-list expansion could also cause
repeated NN evaluations of a resident but unexpanded node. Both behaviors could
increase root Visits and Q while the relevant branch stopped growing.

The revised search checks storage **before** NN inference. A failed attempt
contributes no NN value, completed simulation, root visit, Q backup or depth
sample. The search then reclaims low-visit descendants inside the existing
node/edge reservations and retries from the root.

- Root, every root edge, and statistics on retained internal nodes/edges survive.
- Existing direct root children are protected when that protected set fits both
  half-capacity targets. Tiny node budgets can relax this protection.
- The visit threshold starts at 2 and doubles until retained nodes and retained
  edges each fit half their respective reserved capacities. This creates room
  for both kinds of allocation.
- Marking and index assignment are separate passes. New node IDs follow old ID
  order. Edge blocks are copied in old block order before nodes are compacted.
  The previous traversal path is discarded before retrying.
- The arena is compacted in place; a second complete tree is not allocated.
  Keep flags, remapping and block-order arrays use separately reported scratch.
- If memory cannot be reclaimed, the search returns with `memory_exhausted=true`
  and the actual completed N. It never invents requested visits or loops forever.

This is an approximate bounded-memory MCTS: discarded descendant statistics are
lost and may be rebuilt later. It is not equivalent to unlimited-tree PUCT, and
deeper measured paths alone do not establish greater playing strength.

## Build and use

Visual Studio C++ tools, CMake and Ninja are required to rebuild:

```powershell
.\build.cmd
ctest --test-dir build --output-on-failure
```

Run `.\asan.cmd` for a separate AddressSanitizer build and the same five tests.

Run `dist/chromatic-memory.exe` as a UCI engine. The search fix is always active;
optional detailed telemetry is off by default.

```text
uci
setoption name TreeMemoryMB value 64
setoption name MaxNodes value 1000000
setoption name MateProofNodes value 0
setoption name SearchTelemetry value true
setoption name TelemetryInterval value 4096
isready
position startpos
go movetime 10000
```

For a deterministic completed-simulation budget and explicit seed:

```text
position startpos
zero search 722781 2026 0 0
```

The native probe separates search wall time from model load, warmup and final
JSON transmission. All comparison runs use one search thread, CPU affinity mask
2, SIMD/sparse inference, CPuct 1.5, FPU 0, proof/noise/temperature 0 and seed 2026.

```powershell
.\dist\depth_probe.exe --model .\dist\default.zero --mode time --budget 10000 --memory 64 --max-nodes 1000000 --seed 2026 --threads 1 --affinity 2 --telemetry true --interval 4096
.\dist\depth_probe.exe --model .\dist\default.zero --mode simulations --budget 722781 --memory 512 --max-nodes 1000000 --seed 2026 --threads 1 --affinity 2 --telemetry true --interval 4096
```

The same MaxNodes limit is used at both byte budgets so it does not hide the
512MiB comparison behind the old 100,000-node default. The UCI default remains
100,000 unless explicitly set.

## Counters and memory definitions

Telemetry schema version 2 retains the depth histograms, candidate histories and
first-failure events of schema 1 and adds recycling records.

- `simulations` and each root's Visits count only completed backups. Outside the
  separate proof-only path, `attempted_simulations - aborted_simulations == simulations`.
- `node_creations` is cumulative across recycling. Therefore
  `node_creations + 1 - nodes_evicted == arena_nodes`.
- `legal_expansions` is cumulative successful legal-edge-list storage, including
  `root_legal_expansions`. It is different from node creation and NN calls.
- `non_retained_neural_evaluations`, `resident_unexpanded_repeat` and
  `discarded_neural_evaluations` should be zero in this version. Failed storage
  is detected before inference.
- `prune_count` counts successful reclamations. Each `prunes` record stores
  before/after sizes, threshold, duration, scratch bytes, root child protection,
  and exact root Policy/Q/Visits preservation. An unsuccessful reclamation can
  be recorded without increasing the successful count.
- Failure events identify both the attempt and completed N. Their `simulation`
  is completed N + 1. If time expires before retry, it can equal final N + 1.
- A recycle snapshot may have no new depth samples: completed Visits did not
  change. Empty windows must be skipped rather than plotted as depth zero.
- All depths are plies below the current root. One completed simulation creates
  one endpoint sample. Traversal depth d is the tail sum of endpoint depths ≥d.
  Root initialization, failed attempts and mate proof are excluded.
- `memory_bytes` and `peak_memory_bytes` are the node/edge reserved-capacity
  total and its maximum. The 64MiB/512MiB limits apply to that total. They are
  not process peak memory and exclude transient `vector::reserve` allocation,
  model/workspace, telemetry, serializer and algorithm scratch.
- `peak_auxiliary_bytes` separately reports the keep/remap/block-order scratch
  peak. `telemetry_memory_bytes` estimates the retained diagnostic result only.

Detailed telemetry JSON is produced after search and before `bestmove`, so UCI
delivery time includes formatting/transmission when enabled. Probe search time
includes pruning and final result cleanup. It excludes model load, 32 warmup NN
calls and JSON serialization. Actual process time is recorded separately.

## Verification and experiment data

Five native tests pass both normally and with AddressSanitizer: existing core,
uniform/model depth accounting, and uniform/model recycling regression tests.
They cover repeated reclamation, root statistics, off/on equivalence, stale ID
flags, no-progress termination, repetition history, promotion and en passant.

The sibling `chromatic-memory-analysis` directory contains contemporaneous
legacy/revised measurements, tables, plots and raw logs. Time comparisons and
equal-completed-N comparisons are kept separate. The model and prior releases
are verified by SHA-256 before and after the batch.
