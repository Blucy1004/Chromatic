# Performance evidence — Operation Overlord

All new features remain **off by default**. The original release/champion files are preserved. The strongest verified result is better exact first-reply selection on the existing official puzzle sample. These games do not establish an Elo gain.

## Common conditions

- Linux x86_64, GCC 14.2 Release (-O3, no fast-math), C++20. Two allowed CPU IDs are recorded in each experiment metadata file; no builds or trainer tests ran during matches.
- Both sides load the identical project-chromatic-memory/dist/default.zero; SHA-256 and binary hashes are in every metadata.json. SIMD/sparse policy, CPuct 1.5, FPU 0, noise/temperature 0, seed 1; original mate proof disabled for ablations to isolate the modules.
- Serial arms each use one search thread. Parallel games compare two shared-tree search threads against two independent legacy-algorithm search trees, with the same total 64 MiB tree budget. This control is explicitly different from the historical single-thread executable.
- Fixed 30ms per move on four paired starting positions (two opening lines, two endings). A separate 100ms test uses Sicilian/French/English/Slav lines. Colors swap; results, exact moves, per-move wall times and MCTS/verification counters are retained.
- No resignation, score-based draw adjudication, fake outcomes or champion promotion. Repetition/50-move claims use python-chess claim_draw=True. Games reaching 240 searched plies remain unfinished.

## Official puzzle first replies (final engine)

100 existing CC0 Lichess puzzle records; apply the published opponent move, then compare the engine move with the first published reply. The positions were not used to tune these modules. They are a small fixed prefix sample, and historical champion training provenance is unknown. Different moves can sometimes also win. This is not an exhaustive tactical oracle or a rating test.

| Configuration | ms/position | Correct / 100 | Wilson 95% interval |
|---|---:|---:|---|
| disabled | 30 | 21 | 0.142–0.300 |
| counterexample | 30 | 25 | 0.175–0.343 |
| adaptive | 30 | 23 | 0.158–0.322 |
| forcing | 30 | 47 | 0.375–0.567 |
| tactical | 30 | 46 | 0.366–0.557 |
| parallel | 30 | 21 | 0.142–0.300 |
| combination | 30 | 36 | 0.273–0.458 |
| disabled | 100 | 22 | 0.150–0.311 |
| tactical | 100 | 45 | 0.356–0.548 |
| combination | 100 | 56 | 0.462–0.653 |

The development-only 8-position diagnostic improved from 3/8 disabled to 7/8 with tactical verification. That diagnostic was used during development and is not holdout evidence. Official-puzzle raw results and hashes: [JSON](reports/puzzles-final.json), [CSV](reports/puzzles-final.csv).

## Full feature ablation, second iteration (30ms)

| Configuration | W–D–L | Unfinished | Diagnostic / 8 |
|---|---|---:|---:|
| disabled | 0–8–0 | 0 | 3 |
| counterexample | 0–6–1 | 1 | 3 |
| adaptive | 0–8–0 | 0 | 3 |
| forcing | 1–7–0 | 0 | 4 |
| tactical | 2–6–0 | 0 | 7 |
| parallel | 0–8–0 | 0 | 3 |
| combination | 1–6–0 | 1 | 6 |

This iteration admitted low-prior captures/promotions/checks after the first iteration exposed shortlist failures. The serial search paths are the final design. The final revision changes only parallel worker accounting; its separate results below supersede the earlier parallel arm. Scores that omit unfinished games are descriptive and cannot support promotion.

Raw data: [second iteration](reports/ablation-v2-30ms/summary.json), [performance CSV](reports/ablation-v2-30ms/performance.csv). The [first 56-game iteration](reports/ablation-30ms/summary.json) remains intact, including poor forcing/tactical outcomes.

## Final two-thread comparison

- disabled: 0 wins, 8 draws, 0 losses; median 1431.5 simulations in 30.24ms across 12 balanced position searches.
- parallel: 0 wins, 8 draws, 0 losses; median 1610.0 simulations in 30.24ms across 12 balanced position searches.

The corrected shared-tree engine uses Threads−1 helpers and runs one leaf on the coordinator. Threads includes that coordinator. The two-thread arm completed more simulations in this small measurement, but puzzle accuracy and games did not improve. Earlier helper-plus-coordinator versions showed variable or worse throughput and are retained. This is not evidence to raise the default thread count. Peak in-flight fields count reserved leaves awaiting commit, not an instrumented count of simultaneous CPU instructions.

## Independent opening lines at 100ms

Final tactical arm: 1 win, 7 draws, 0 losses across 4 color-swapped opening pairs. Final combination arm: **5 wins, 2 draws, 0 losses, 1 unfinished**; only 3 complete opening pairs. Its conservative paired-score interval is **[0.049, 1]**. Both raw arms are retained in reports/holdout-tactical-100ms and reports/holdout-combination-100ms. Both use the same model and one search thread per side.

For the tactical arm with only four complete opening pairs, the conservative 95% Hoeffding interval is [0,1]. The incomplete combination pair is excluded from its pair-based interval. Repeated runs of these same openings must not be counted as additional independent opening samples. No Elo estimate or promotion decision is justified. Puzzle Wilson intervals quantify uncertainty within that fixed sample only; they do not predict game Elo.

## Selection and retained artifacts

- Counterexample/adaptive modules have weak small-sample signals and stay off. Short forcing search regressed in the initial iteration; candidate admission corrected clear misses, but finite horizons can still miss quiet defenses.
- Tactical verification is a promising opt-in candidate: stronger published-reply accuracy and favorable small matches. It introduces a bounded material/WDL heuristic, so strategic and sacrifice regressions remain possible.
- The all-feature combination needs enough time to finish all candidate comparisons; incomplete comparisons preserve the MCTS decision. Its 30ms game arm contained an unfinished game. It stays off.
- Every directory has PGN plus compressed raw per-game JSON (.json.gz), exact options, binary/model hashes, CSV summaries and per-position benchmark JSON. Run tools/summarize.py on a directory to replay/check its PGNs and regenerate aggregates.
- Follow-up before any stable promotion: a much larger unique opening suite, longer clock controls, multiple seeds/runs, separate sacrifice/endgame tests and Windows/MSVC runtime validation. Existing binaries and champions remain the default release.

## Overall run accounting

144 games were recorded across both development iterations and final controls:
141 finished and 3 remained unfinished at the ply cap. Every PGN and compressed
raw trace was replayed/validated, with no engine errors or illegal moves. These
144 games contain repeated openings and different candidate revisions; they must
not be pooled into a 144-game independent Elo estimate. The official puzzle
check made 1,000 searches over the same 100 positions under ten configurations /
time settings; this is 100 unique puzzles, not 1,000 independent examples.
