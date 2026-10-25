# PROJECT CHROMATIC Hybrid Implementation Contract

## Source baseline and isolation

This folder is an isolated copy of the 1.1 memory-recycling source plus the pre-existing ValueZero diagnostic experiment source. Source basis: repository commit `5310949551fd76d8a6702daf2ddeb996fd5f3c9b`; the source copy already includes the experiment-only ValueZero and fallback telemetry edits. The existing MCTS implementation, `baseline/`, original model `dist/default.zero`, old release packages, match PGNs and logs remain untouched elsewhere. Do not change the default engine mode or champion/model.

This checkout was distributed as a source folder, not a Git repository; the implementation is isolated by directory name `project-chromatic-hybrid`, not by an invented remote or destructive branch reset. Hash/source provenance will be recorded with deliverables.

## Ownership / integration

- Search/evaluation: `include/zero/pvs.hpp` (integrator contract); agent A owns `src/pvs.cpp` only.
- RRN reference module: agent B owns `include/zero/rrn.hpp` and `src/rrn.cpp` only.
- Regression tests/harness: agent C owns new `tests/hybrid_tests.cpp` only.
- Integrator owns `src/main.cpp`, `CMakeLists.txt`, run scripts, docs, package, and integration fixes. No worker edits those files.

## PVS contract

`PvsSearch::run` accepts an immutable root Board, explicit limits and a stop flag. Scores are centipawns from root side-to-move; mate scores are `+/- (PVS_MATE_SCORE - ply)`. It must preserve the Board by searching private copies or balanced make/unmake and obey node/time/stop limits. Iterative deepening returns only the last fully completed iteration. TT entries must verify the full key, depth, bound type and halfmove/history safety; collisions may replace but never return an unverified score. Ordering hints are scores parallel to the legal move vector and can only reorder it. No hint may remove a legal move.

## RRN contract

RRN uses typed board nodes/relations, tied message-passing weights across iterations and a per-legal-move score head. The module must report whether weights are learned or deterministic untrained reference. Missing/invalid weights must not silently be treated as learned. Default UCI mode is off. Scores only reorder legal candidates.

## Follow-up extensions

TRM selective candidate refinement and Set Transformer compute-allocation remain interface/design only in this milestone; neither is implemented or active. Learned RRN weights and model training are separate from deterministic inference/reference and require their own data/provenance/heldout evidence.
