# Validation contracts and evidence

The final source is an experimental branch; existing deployments and champions
are unchanged. Linux GCC 14.2 / C++20 / CMake 4.1.3 / Ninja 1.13.0 were used.
Python 3.12 / PyTorch 2.9.1+cpu run the existing integration/trainer checks.

## Required checks executed

| Check | Outcome | Evidence |
|---|---|---|
| Historical 1.0 native regression | 1/1 passed | reports/original-native.log |
| Historical depth native regression | 3/3 passed | reports/depth-native.log |
| Unmodified memory-recycling baseline | 5/5 passed | reports/baseline-native.log |
| Candidate Release native regressions | 7/7 passed | reports/release-final.log |
| AddressSanitizer + UndefinedBehaviorSanitizer | 7/7 passed | reports/asan-final.log |
| ThreadSanitizer with actual loaded model | passed | reports/tsan-final.log |
| Existing Python model/oracle/UCI/training integration | 5 passed | reports/integration.json, integration.log |
| Existing replay, holdout and tiny-data training | 3 passed | reports/reliability.json |
| Existing invalid-model/protocol/checkpoint safety | 4 passed | reports/safety.json |
| New black-box UCI/model/history checks | passed | reports/uci-final.json |
| ARC trainer existing regression, disk-backed temp | 149 passed, 24 subtests; 8 skipped | reports/arc-disk-regression.log |
| ARC actual kill/resume suite on isolated copy | 6 passed | reports/arc-recovery.log, arc-recovery.json |
| Preserved CPU shard trainer regression | 57 passed, 20 subtests; 7 skipped | reports/shard-regression.log |
| Experiment accounting/history preservation tests | 4 passed | reports/experiment-tests.log |

The eight ARC skips are six opt-in process tests (executed separately above), one
real XPU test and one Windows PowerShell test. The older CPU trainer's six opt-in
process tests and Windows PowerShell integration remain skipped in its suite;
ARC recovery evidence is not relabeled as a run of the older suite.

## What can fail these checks

- Perft across five reference positions; independent Python legality/features
  oracle over 1,008 positions; repeated-position and pinned-EP identities.
- Synthetic WDL weights enforce a child win becoming a parent loss, at Threads
  1/2/4. Uniform-policy mate tests isolate backup sign from learned root policy.
- Tiny node/byte budgets force repeated compaction; before/after root priors, Q
  and visits must agree exactly. Abort counts cannot become simulations. Parallel
  root N sums equal real completed simulations after virtual losses disappear.
- Parallel NN workers use separate workspaces. Telemetry must account for every
  endpoint and forbid resident unexpanded repeat evaluation. TSan executes the
  same constrained-tree tests with the deployed model. This is evidence for the
  exercised schedules, not a proof covering every possible interleaving.
- Mate, stalemate, insufficient material, threefold and 50-move roots are
  terminal before heuristic verification. Candidate branch boards retain history.
- A one-node incomplete comparison cannot change the MCTS decision. Synthetic
  hanging-queen tests require admission of low-visit captures. Exact root mate
  scans must prove a real terminal board independently of network policy.
- Timed/infinite search and stop join all workers; large time values exercise
  deadline arithmetic without signed overflow. Timed validation records actual
  delivery latency, including output. Wave drain, legal generation and one NN
  evaluation are not preemptible, so the deadline is cooperative rather than an
  OS-level hard real-time guarantee.
- Existing model tests compare full versus sparse policy and scalar versus SIMD;
  invalid model loads preserve the old model and error paths keep UCI responsive.
- Game validation replays PGNs, checks legal moves and result consistency against
  python-chess. Ply-limited games stay `*`, never invented draws. Color pairs are
  statistical units, and tiny samples never promote a candidate.

## Failure history retained

reports/release-first.log: a test conflated learned low-prior exploration with
backup correctness; its corrected separation is explained in AUDIT.md.
reports/integration-first.log: Linux copyfile removed test executable permission;
copy2 fixes only the candidate test fixture.
reports/arc-regression.log: physical I/O bytes are zero on /tmp; controlled child
experiments and the passing disk-backed rerun are both retained.
First-generation and second-generation ablations are both preserved, including
weak performance, missed tactics and unfinished games. No assertion, verification
or failed game was removed to manufacture a passing result.

## Windows compatibility review

The candidate retains C++20, the ZERO0001 FP32 format, SSE2/Scalar dispatch and
standard-library threading. It introduces no POSIX APIs in engine C++ code.
CMake retains explicit MSVC /W4 /permissive- /fp:precise and an ASan option.
Use a Visual Studio developer shell with CMake and Ninja, pass the existing
model explicitly, and run CTest before packaging a Windows candidate. Python
CPU-affinity use is platform-guarded in experiment scripts. Windows execution,
MSVC compilation and Windows ASan runtime loading were **not executed here**;
there is no newly verified Windows binary and no replacement of the old one.
