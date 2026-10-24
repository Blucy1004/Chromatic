# CHROMATIC optional MCTS depth telemetry

This is a separate diagnostic build of CHROMATIC 1.0. The existing
`outputs/project-chromatic` release and its model are preserved. No training,
weight changes, search-strength changes, allocator fixes or move filters are
included. `dist/default.zero` is a byte-identical copy of the deployed model.

Build on Windows with Visual Studio C++ tools, CMake and Ninja:

```powershell
.\build.cmd
ctest --test-dir build --output-on-failure
```

The executable `dist/chromatic-depth.exe` speaks UCI. **Telemetry is off by
default.** Turn it on when collecting diagnostics:

```text
uci
setoption name SearchTelemetry value true
setoption name TelemetryInterval value 4096
setoption name TreeMemoryMB value 512
setoption name MaxNodes value 1000000
setoption name MateProofNodes value 0
isready
position startpos
go movetime 1000
```

UCI `go` emits the usual search summary, then an `info string telemetry {JSON}`
line, then `bestmove`. JSON serialization/transmission takes place after search
and can add response-delivery time when telemetry is on. Use the probe for
separately measured search and process times. With telemetry off, neither the
histogram bookkeeping nor telemetry JSON is produced.

The `zero search` extension accepts an explicit seed and returns the same JSON
payload, including telemetry only when enabled:

```text
position startpos
zero search 722781 2026 0 0
```

For controlled time and simulation budgets use `dist/depth_probe.exe`. All
experiments use one search thread, CPU affinity mask 2, SIMD, sparse policy,
CPuct 1.5, FPU reduction 0, noise/temperature/proof 0, and seed 2026. Search is
fresh for every invocation. `--max-nodes 1000000` is the same at both byte caps.

```powershell
.\dist\depth_probe.exe --model .\dist\default.zero --mode time --budget 1000 --memory 64 --max-nodes 1000000 --seed 2026 --threads 1 --affinity 2 --telemetry true --interval 4096
.\dist\depth_probe.exe --model .\dist\default.zero --mode simulations --budget 722781 --memory 512 --max-nodes 1000000 --seed 2026 --threads 1 --affinity 2 --telemetry true --interval 4096
```

`--fen` accepts a quoted six-field FEN. `--telemetry false` uses the disabled
template instantiation. `--warmup true` performs 32 NN evaluations before the
search timer. `elapsed_ms` measures the search call, including final telemetry
preparation and tree cleanup, but excludes model loading, warmup and JSON
serialization. Experiment logs separately record process wall time.

`pristine_probe.exe` links the unmodified source in `baseline/`, with the same
compiler and optimization options, and is used only for overhead comparisons.

## Metric definitions

- Root depth is 0. Each completed MCTS simulation contributes one endpoint
  depth measured in plies below that root. It may be a newly expanded leaf,
  a cached terminal, or an unretained NN-evaluated leaf after allocation failure.
  It is not necessarily a unique position.
- `leaf_depth[d]` counts endpoints at exactly depth d. Mean and median are exact
  visit-weighted statistics of this histogram, not sampled estimates.
- `traversal_depth[0]` is the number of completed MCTS simulations.
  `traversal_depth[d]` for d >= 1 counts traversed edges at depth d and equals
  `sum(leaf_depth[d:])`. These are path visits, not distinct nodes at depth d.
- Each root candidate has Policy (`prior`, normalized over legal moves), Q
  (average backed-up W−L from the root player's perspective), Visits, and its
  own conditional endpoint-depth histogram. Q is not a win probability or a
  centipawn evaluation.
- `node_creations` counts newly retained nodes and excludes root initialization.
  `legal_expansions` counts successfully stored legal-edge lists, including
  `root_legal_expansions`. Thus newly created terminal nodes need not count as
  legal-edge expansions.
- Child creation failures are split into node-limit and byte-capacity failures.
  `expansion_fail_byte_capacity` counts legal-edge-list storage failures.
  Their sum counts failed allocation attempts, not unique failed positions.
- `non_retained_neural_evaluations` counts NN calls whose evaluated leaf has no
  stored node. It does not count exact repeated NN inputs. Resident-unexpanded
  repeats are tracked separately.
- The primary saturation event is `first_allocation_failure`, with a 1-based
  simulation index, elapsed time, attempted depth, sizes and reserved capacities.
  The first failing simulation belongs to `post_failure_leaf_depth`; earlier
  simulations belong to `pre_failure_leaf_depth`.
- `first_reserved_cap` reports the separate event where reserved bytes exactly
  equal the byte budget. Integer-division slack can prevent this event even when
  later allocation fails; use actual failure to identify saturation.
- Snapshots are captured at 64, 128, ..., 4096 simulations, each configured
  interval, first capacity events, and completion. Events are taken during the
  allocation attempt; snapshots describe the state after that simulation's
  completed backup. Candidate crossover times are interval-resolved, not exact
  per-simulation timestamps. Snapshot windows partition the full histogram.
- Root initialization is not a simulation endpoint. Terminal-root searches have
  zero endpoints. A successful bounded mate proof retains the old one-root-update
  result but is explicitly `proof_only` with zero MCTS endpoints.
- The inherited path safety limit is 512 resident path nodes, or at most 511
  plies below root. Depth-limit endpoints are counted. None of these metrics
  establishes proven minimax depth or playing strength.
- Tree budgets apply to Node/Edge reserved capacities. Telemetry is kept outside
  this budget. `telemetry_memory_bytes` estimates the retained result structure,
  not peak process memory; temporary flags and serializer buffers are additional.

## Validation and results

Native tests cover perft, backup sign, off/on exact search equivalence, histogram
accounting, exact medians, root partitions, snapshot windows, capacity failures,
terminal roots and proof-only returns. The disabled hot loop is selected once
through `search_impl<false>`; `if constexpr` removes counter/snapshot work.

The sibling `chromatic-depth-analysis` directory contains the complete report,
tables, PNG/SVG figures, JSONL runs, per-run requests/stdout, settings, integrity
hashes and separate randomized overhead results. Time-budget results have three
repeats. Fixed-simulation results use equal N at both memory budgets and are
reported separately. See its report for measured performance limits and findings.
