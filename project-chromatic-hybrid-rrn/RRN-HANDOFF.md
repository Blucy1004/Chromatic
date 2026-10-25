# RRN training and inference handoff

Snapshot: RRN hybrid source package based on the local CHROMATIC hybrid project derived from source commit 5310949551fd76d8a6702daf2ddeb996fd5f3c9b; the hybrid/RRN files themselves are not in the Git trees available in the local clone, so this is a source snapshot, not a Git commit. No Git metadata or secrets are included.

## RRN model and interface

- RRN node graph has 64 board-square nodes, 19 features per node, hidden width 8, four relation types, and three shared message-passing steps. The relations are adjacent squares, same-color pieces, same-type pieces, and legal-move candidate edges.
- Legal moves are scored by a learned ranking head using source/destination node states plus five move features. Hints only alter move ordering and never prune a legal move.
- Weights format header: CHROMATIC_RRN_WEIGHTS 1 19 8 4 3; then learned flag, seed, provenance and the named flattened weight tensors. Python export and C++ load/save live in 	ools/train_rrn.py and src/rrn.cpp.
- Existing RRN weights SHA-256: $weightHash; default neural sidecar for UCI startup SHA-256: $modelHash. The neural sidecar is needed for the packaged UCI executable startup, but PVS evaluation uses explicit material/PST; it is not the RRN weights.

## Existing training evidence

- Existing teacher data is 32 deterministic random-legal trajectories, 24 sampled positions each, split by trajectory: 576 train and 192 heldout. Teacher: Stockfish 19, 1 thread, 16 MB hash, 1,024 nodes per position. This is not 1,000,000 positions and not Lichess data.
- Existing model training: seed 20261025, Adam, learning rate .001, batch 32, 2 CPU threads, exactly 200 steps (about 5 seconds). Metrics are in 	raining/rrn-trained.metrics.json; heldout ranking top-1 was 38.54%, MRR .5182. These are teacher move-imitation metrics, not playing strength.
- Re-run training: python tools/train_rrn.py --data training/rrn-teacher-dataset.json --out training/rrn-trained.weights --steps 200 --wall-seconds 120 (requires Python with NumPy, PyTorch and python-chess).
- Regenerate teacher rows only with an authorized Stockfish path and explicit fixed budget: python tools/generate_rrn_data.py --help.

## Validation / build

- Build UCI engine: cmake -S . -B build -DCMAKE_BUILD_TYPE=Release then cmake --build build --config Release.
- C++ regression tests and RRN ranking smoke sources are included under 	ests/ and 	ools/rrn_weight_smoke.py; the original fixed CHROMATIC baseline remains outside this handoff ZIP.
- No new training, games, upload to GitHub or deployment were performed as part of this handoff preparation.
