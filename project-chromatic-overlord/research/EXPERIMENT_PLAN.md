# Predeclared factorial protocol

All existing deployments, champions and earlier evidence are preserved. No model
promotion or default activation is allowed by this protocol.

## Factors and controls

Bits 0..4 are C=CounterexampleVerify, A=AdaptiveBudget, F=ForcingSearch,
T=TacticalVerify, P=Threads 2 instead of 1, SharedTree true. Thus the actual
four boolean algorithms produce 16 serial configurations, supplemented by the
same 16 configurations with the parallel resource factor. All 32 are valid.
P includes extra CPU concurrency and is not a free algorithm-only intervention.
A single-thread baseline is mask 0; four-module full serial is 15; full with
parallel is 31. AEGIS is absent. tactical_safety is unused. MateProof is an older,
nonbinary proof-budget control, held at zero for the factorial and measured
separately at 256 nodes. Root-ensemble SharedTree=false is not a factorial factor:
it rejects verification/telemetry at Threads>1 and is explicitly excluded.

Models: initial ZERO128, plus canonical positions-100/1000/10000 candidates.
All architectures remain identical; data scale is not parameter count.
Same seed 1, SIMD sparse policy, CPuct 1.5, FPU 0, 64MiB arena, MaxNodes 100000,
verification cap 2048 nodes / 4 plies / 4 candidates, no noise/temperature.
Every cell records model/tensor/executable/source hashes, complete options, OS,
hardware identity, CPU affinity and wall/CPU cost. Driver uses a spare core when
available; search is always restricted to the same two CPU IDs on each host.

## Phase budget (each OS, one stable host)

| Phase | Model/config/position scope | Nominal search seconds |
|---|---|---:|
| Screen | 4 × 32 × old 100, 100ms | 1280 |
| Legacy proof control | 4 × old 100, 100ms | 40 |
| Repeats | 2 core × up to 4 anchors × 100 × 2 extra, 100ms | ≤160 |
| 1 second | 4 models × 8 arms × fixed rating-stratified 24 | 768 |
| 5 seconds | 2 core × up to 4 arms × fixed stratified 16 | ≤640 |
| 10 seconds | 2 core × up to 3 arms × 4 gain/loss diagnostic cases | ≤240 |
| Fixed visits / total NN cap | 4 × 32 × 24 per mode, budget 512 | estimate 120–480 |
| Independent holdout | 4 × up to 6 locked arms × 200, 100ms | ≤480 |
| Games | 2 core selected serial arms + optional ZERO full, 8 pairs/arm, 100ms/move | ≤960 |

Expected 60–80 minutes per platform including diagnostics and output. The runner
stops after 5400 active wall seconds, preserves its journals and reports any
unrun cells. Native CI has a separate 120-minute job ceiling. Linux and Windows
run concurrently on different hosts; only within-host contrasts are causal
controls. No claim that different hardware isolates an OS-only effect is allowed.

## Selection lock and untouched evaluation

The old 100 puzzles are the development/screening set. The next 200 official
puzzles were frozen before screen outcomes, excluding old puzzle IDs, source game
IDs, duplicate normalized positions, and exact positions in the available eval
training-input superset and bootstrap corpus. No outcome on this holdout is used
for selecting configurations. The bounded upstream capture has provenance/TLS
and a prefix hash; full remote Zstandard frame checksum is not claimed.

Linux screen chooses best serial per model: most hits, then less total CPU,
then fewer flags, then numeric mask. It locks one primary *serial* pair by the
largest summed zero128/10k pair interaction. 1s controls include 0/C/A/F/T/15 plus
highest-ranking other arms, up to 8. The 5s set adds the most negative screen pair.
10s cases are explicitly diagnostic gain/loss cases, not an unbiased score set.
Windows downloads that same Linux plan and never retunes it on Windows results.
Holdout includes baseline, serial full, selected serial and the chosen pair's
A/B/AB conditions. Games use screen-selected serial configurations, not a second
selection on the holdout.

## Estimands and uncertainty

Screen: paired puzzle-level accuracy, gains/losses, full-minus-one, 5-factor
Shapley and separate 4-factor Shapley conditional on P. All 10 pair interactions
and higher-order Möbius contrasts are exploratory on the screen. Shapley is
conditional on this metric, factor set, budget, and corpus, not intrinsic value.

Report paired/cluster bootstrap intervals, exact McNemar for binary contrasts,
Holm family correction, and repeat variation without counting repeats as new
puzzles. Holdout primary pair interaction uses paired bootstrap and simultaneous
intervals across 4 models. A positive performance synergy requires positive
corrected interval *and* corrected gains over both components; otherwise it is
not confirmed. Bootstrap coverage depends on sampling assumptions; this fixed
prefix corpus does not certify general chess strength.

Fixed visit budgets do not equalize verification work. The separate total NN cap
includes root, tree and verifier, reserves 25% for verification, and has a 16×
simulation safety cap for cached terminals. A cap is not necessarily fully spent;
actual work and early stops are reported. Non-NN move generation remains extra
CPU work. Neither cap is misrepresented as exactly equal CPU instructions.

Games: original outcomes only, no loss-score adjudication, color-swapped unique
opening lines, PGN and all moves retained. Unfinished games remain unfinished.
Report descriptive Elo and pair-bootstrap uncertainty only if complete; promotion
requires at least 64 independent opening pairs plus a score lower bound above
half. This bounded experiment has fewer pairs and cannot confirm an Elo gain.

## Durability

Cell journals fsync every puzzle, compress only after completion, and bind every
resume to the same executable/models/source/runner/corpus/host/CPU affinity.
Corrupt middle records abort; only an interrupted final record may be recovered
with its tail preserved. An active process lock prevents concurrent writers.
New CI hosts are separate blocks; evidence from different hosts is not silently
pooled. Stage events retain nominal and actual work, failures, pauses and skips.
