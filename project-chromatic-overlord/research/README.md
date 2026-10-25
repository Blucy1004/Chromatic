# Reproduce the factorial research

No promotion or default activation. Read [protocol](EXPERIMENT_PLAN.md),
[module audit](MODULE_AUDIT.md), and [selection lock](selection-plan.json).
The old100 puzzles select configurations; the frozen200 evaluate them.
Exact expected-response matches are not Elo.

## Dependencies and native build

Python3.12, chess1.10.0 / python-chess1.999, psutil7.2.2 for execution;
NumPy2.4.6 for statistics. Figures use system Python with Matplotlib3.10.8 /
NumPy2.3.5, without recomputing statistics. CMake3.20+, C++20, threads.
Compiler records, CPU/affinity and executable/source/model hashes accompany runs.

```sh
cmake -S project-chromatic-overlord -B /tmp/chromatic-factorial -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/chromatic-factorial --parallel 2
ctest --test-dir /tmp/chromatic-factorial --output-on-failure
python project-chromatic-overlord/research/test_statistics.py
```

Windows: `overlord-factorial.yml` builds native MSVC x64 Release and executes
all CTests and the same runner. `factorial-artifact-validation.yml` downloads
that measured executable, checks its digest, then performs native UCI, stop/quit,
time, reclamation/perft and research-protocol checks on a separate Windows host.
That host is protocol evidence, not a performance replicate.

## Data and preflight

Development: `chromatic-arc-trainer/fixtures/official-puzzles.csv`,100 puzzles.
Untouched: `research/fixtures/holdout.csv`,200 puzzles, frozen before outcomes
by `collect_holdout.py` and the data-freeze workflow. Source manifest documents
the HTTPS compressed prefix and does not claim the whole remote frame checksum.
`preflight.py` validates four PT exports, actual C++ tensor digests, data
separation, interrupted-tail recovery, unique resume rows and writer exclusion.
`calibrate.py` checks old-engine and metrics ON/OFF fixed-work parity, plus
observer cost. Retained raw checks: `reports/factorial/preflight`.

## Run and resume

Use a fresh output directory for a new host/build; one driver per host. The
driver locks two CPU IDs for engines and a spare CPU for itself. Avoid other
work on search CPUs. `--publish-plan` explicitly publishes the Linux-selected
plan and requires a clean git staging area. Windows retrieves that same plan
and validates runner/C++/corpus hashes, without retuning on Windows outcomes.

```sh
python project-chromatic-overlord/research/run.py --engine /tmp/chromatic-factorial/chromatic_overlord --out project-chromatic-overlord/reports/factorial/linux/NEW-BLOCK --stage pipeline --max-seconds 5400
```

Rerun the identical command to resume on the same host, executable, models,
runner, C++ source, CPU IDs and corpus. Elapsed budget persists. Exit75 is paused,
not successful. `completed.json` and final pipeline event mark completion.
`--stage followup --plan project-chromatic-overlord/research/selection-plan.json`
uses the retained locked plan on compatible source. A new source requires a new
block/plan. Source commit alone is insufficient: compare actual hashed contents.

Journals flush+fsync each puzzle. Only an invalid last row is recoverable, with
its bytes retained; middle corruption aborts. Completed cells bind model and
binary closing attestations to compressed hashes. Abrupt kill during the
compression/metadata multi-file handoff may need manual review/new block.
Never overwrite or silently pool different-host evidence.

Standard asynchronous UCI `go` supports stop/quit. `zero bench time|nodes|nn
BUDGET SEED` is synchronous and bounded, without interleaved stop. The runner
uses short calls, checkpoints between puzzles and saves progress immediately.
Total NN cap includes root+MCTS+verifier and need not be fully spent.

## Analysis without engine execution

Run from repository root. Replace Windows ID if reproducing a different block.
Analysis verifies closing identities and compressed hashes before ingesting.
Use a spare CPU (`taskset -c 3`) and `OPENBLAS_NUM_THREADS=1` if search is active.

```sh
python project-chromatic-overlord/research/analyze.py --run project-chromatic-overlord/reports/factorial/linux/20261025-v1 --run project-chromatic-overlord/reports/factorial/windows/40152066905-1 --out project-chromatic-overlord/reports/factorial/analysis --bootstrap 5000
python project-chromatic-overlord/research/cross_analysis.py --linux project-chromatic-overlord/reports/factorial/linux/20261025-v1 --windows project-chromatic-overlord/reports/factorial/windows/40152066905-1 --out project-chromatic-overlord/reports/factorial/analysis --bootstrap 5000
python project-chromatic-overlord/research/match_analysis.py --run project-chromatic-overlord/reports/factorial/linux/20261025-v1 --run project-chromatic-overlord/reports/factorial/windows/40152066905-1 --out project-chromatic-overlord/reports/factorial/analysis
python project-chromatic-overlord/research/plot.py --analysis project-chromatic-overlord/reports/factorial/analysis --plan project-chromatic-overlord/research/selection-plan.json
python project-chromatic-overlord/research/report.py --analysis project-chromatic-overlord/reports/factorial/analysis --plan project-chromatic-overlord/research/selection-plan.json
```

For plotting set MPLCONFIGDIR to a writable directory if needed. Bootstrap seed
is fixed. Outputs include full matrix, per-puzzle correctness/moves/PV/cost,
paired contrasts, all-context marginals, 5/4-factor Shapley, all interaction
orders, model/search and OS/search, primary holdout tests, native numeric probes,
PGN/trace-validated games, PNG/SVG. Raw JSONL.gz is original engine evidence.
Read intervals with sampling, multiple-comparison and selection limitations.
