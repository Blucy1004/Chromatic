# CHROMATIC — Operation Overlord (experimental candidate)

**[Windows x64 Release package and launch instructions](WINDOWS_RELEASE_PACKAGE.md)**

The factorial follow-up is in [FACTORIAL_ABLATION.md](FACTORIAL_ABLATION.md):
model/search contributions, independent puzzle validation, cost, native OS
comparisons and paired games. Reproduce it with [research/README.md](research/README.md).

This separate C++20 candidate starts from the memory-recycling engine at commit
5310949551fd76d8a6702daf2ddeb996fd5f3c9b. Existing engines, deployment binaries,
champions, trainers and weights are preserved. No training or automatic champion
promotion occurs. Every new search feature is opt-in; Threads defaults to 1.

SANABI and Laugh Tale implementations were not found in this repository. These
names describe **new local designs**, not reproductions of external algorithms.

```sh
cmake -S project-chromatic-overlord -B /tmp/overlord -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/overlord -j 4
ctest --test-dir /tmp/overlord --output-on-failure
/tmp/overlord/chromatic_overlord --model project-chromatic-memory/dist/default.zero
```

The engine also builds in a Visual Studio C++20 developer shell using the same
CMake commands. Windows x64/MSVC 19.44 Release and native execution have now passed; see
[WINDOWS_VALIDATION.md](WINDOWS_VALIDATION.md).

The original `default.zero` stays in its existing location. Pass `--model` or
set UCI ModelFile explicitly; the candidate has no bundled new model.

```text
uci
setoption name Threads value 2
setoption name CounterexampleVerify value true
setoption name AdaptiveBudget value true
setoption name ForcingSearch value true
setoption name TacticalVerify value true
isready
position startpos
go movetime 100
```

| Option | Default | Contract |
|---|---|---|
| CounterexampleVerify | false | SANABI-style: all opponent replies to each shortlisted root candidate, pessimistic response score; incomplete comparisons leave MCTS choice intact |
| AdaptiveBudget | false | Laugh Tale-style: root policy entropy and WDL uncertainty adjust root exploration and release time on confident/simple roots; requested time remains a hard ceiling |
| ForcingSearch | false | Bounded quiescence over captures, promotions, checks and all check evasions |
| TacticalVerify | false | Capture/evasion quiescence with bounded material cross-check mixed with unchanged neural WDL |
| Threads | 1 | 1–32 including coordinator; shared-tree synchronous leaf waves, thread-local workspace |
| SharedTree | true | false with Threads>1 selects the independent legacy-tree control; verification/telemetry disabled in that control |
| VerificationNodes | 2048 | Total verification node allowance, divided equally among shortlisted candidates |
| VerificationDepth | 4 | 1–8; finite horizon, not a proof depth |
| VerificationCandidates | 4 | Best MCTS move, admitted forcing alternatives, then visit/prior leaders |

Verification reserves the final 25% of requested movetime. It never alters
MCTS visits, priors or Q; separate counters report its work and changed choices.
If any candidate comparison is interrupted, all tentative scores are ignored.
ForcingSearch and TacticalVerify also scan legal root moves under the same budget
for exact mate-in-one terminals, and admit low-prior captures/promotions (plus
checks for ForcingSearch) to the shortlist. This avoids relying entirely on the
small model's root ranking.
Only terminal values and the retained original MateProof are proofs. Material
and WDL scores remain heuristics. Defaults for original MateProof are retained.

Shared-tree workers perform only read-only neural inference. One coordinator
selects paths, applies temporary virtual loss, installs real results and reclaims
memory after draining a wave. Tree byte limits exclude model/workspaces,
thread stacks, diagnostics and transient queue/path allocations. In-flight work
is bounded by Threads; a synchronous wave may finish after the time check.

## Reproduce experiments

Use the existing checkout; an additional Git worktree is unnecessary. The current
cloud interpreter with dependencies is `/tmp/chromatic-venv/bin/python`.
Dependencies for the runner are python-chess==1.999 (chess==1.10.0 in this run).

```sh
python project-chromatic-overlord/tests/uci_validation.py --engine /tmp/overlord/chromatic_overlord --report /tmp/uci.json
python project-chromatic-overlord/tools/experiment.py --engine /tmp/overlord/chromatic_overlord --baseline /tmp/chromatic-baseline/chromatic_depth --out /tmp/overlord-experiment --ms 30
```

Use a fresh output directory for each run; existing experiments are protected
from overwrite. JSON checkpoints are written atomically. Interrupted experiments
remain inspectable; automatic mid-game resume is not provided.

Build the baseline from the unmodified `project-chromatic-memory` CMake project.
The runner saves model/binary hashes, exact settings, root statistics, per-move
wall times, errors, PGN, CSV and JSON. Openings are color-paired. Two-worker
parallel games use two independent original-algorithm trees as the reference;
that reference is not the historical single-worker executable.

The read-only `zero model` command reports the SHA-256 reconstructed from the
actual loaded ZERO0001 tensors. See [MODEL_PROVENANCE.md](MODEL_PROVENANCE.md)
for the initial 128-position champion and the separate Lichess 10,000-position
candidate, including the platform-specific repeated comparison.

See [AUDIT.md](AUDIT.md), [PERFORMANCE.md](PERFORMANCE.md) and
[VALIDATION.md](VALIDATION.md) for evidence and limits. The preserved deployment
binaries remain the production release; this branch is an experimental source
candidate. The native Windows test binary is retained in the Actions artifact;
it has not replaced the deployed release or been promoted.
