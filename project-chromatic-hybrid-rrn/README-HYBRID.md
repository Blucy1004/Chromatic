# PROJECT CHROMATIC Hybrid

Isolated first vertical implementation. Existing MCTS is retained as a separate mode/executable. The hybrid UCI binary defaults to PVS; RRN is an optional move-order hint and never removes a legal move. The default RRN reference weights are deterministic and untrained; the UCI reports `learned=false`. An optional Stockfish-imitation weights file is included under `training/`, but has not been shown to improve playing strength. TRM and Set Transformer are design-only follow-ups.

Source basis: 1.1 memory-recycling code copied from local candidate evaluation workspace, derived from commit `5310949551fd76d8a6702daf2ddeb996fd5f3c9b`; it includes prior ValueZero diagnostic-only edits in the copied source. No original checkout, package, model, release, or match log was modified. Original `dist/default.zero` SHA-256: `F6A00902C23E7F32F96B028AC36464CC9012CD8774D5E6B0DEE5E1C94E1C03C9`.

No `AGENTS.md` or `.agents/skills` files were present in the accessible workspace tree. Build uses `build.cmd` to initialize Visual Studio x64 tools, configure CMake/Ninja Release, and clean-build; tests use CTest.

