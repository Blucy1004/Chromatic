# RRN learning experiment plan

Status: completed within the fixed budget. The default remains deterministic reference-only; an optional trained imitation candidate was kept as an experiment.

## Resource check

Windows 10.0.26200.0, MSVC 19.51.36248.0. Python environment is isolated at `.venv-train`; PyTorch 2.14.1+cpu, NumPy 2.5.4, python-chess 1.11.2 installed from official CPU wheel/PyPI. `torch.cuda.is_available()` was false; no NVIDIA tooling was found. The PVS-vs-MCTS fixed match was single-concurrency and finished before training began. No paid services or credentials are used.

## Fixed candidate budget

Executed after the fixed engine match. Data came from 32 deterministic random-legal trajectories rather than chess self-play: 24 train and 8 heldout trajectories, up to 24 positions each; FENs were deduplicated globally and split by trajectory. Stockfish 19 ran one thread, Hash 16 MB and 1,024 nodes per position. Counts: 576 train / 192 heldout. Candidate initialization was deterministic-reference-v1; all RRN layers were trained for exactly 200 Adam steps, batch 32, 2 CPU threads, seed 20261025, lr 0.001, within the 120-second budget (about 5 sec). Heldout metrics: reference top-1 13.54%, MRR .3077; candidate top-1 38.54%, MRR .5182. This is only teacher-best-move imitation on synthetic legal trajectories. No heldout tuning, RRN match, or chess-strength inference was performed. Optional candidate weights remain OFF by default.

## Stop conditions

Data generation began only after the fixed engine match. Python-chess SimpleEngine handshake stalled in this Windows venv; a small direct UCI subprocess driver successfully performed the same Stockfish protocol, and its exact option/score/node handling is recorded in `tools/generate_rrn_data.py`. C++ loaded the trained candidate and reported `learned=true`; a legal bestmove smoke passed. Existing `dist/default.zero` remains unchanged.
