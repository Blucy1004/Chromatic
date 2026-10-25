# PROJECT CHROMATIC Hybrid RRN handoff

This folder contains the exact source, tests, training utilities, training metrics, and existing trained RRN weights from the approved handoff archive. The archive SHA-256 is `694FAE38B3979ECAC679749026BA204B54EE731565245E7C5D5261B7ACF6EBBF`. Source bytes are unchanged; see `SOURCE-SHA256SUMS.txt`.

The project is licensed GPL-3.0-or-later (`LICENSE`). Third-party dependencies and Stockfish teacher attribution are listed in `THIRD_PARTY.md`. No external engine source was copied into this project.

`training/rrn-trained.weights` SHA-256: `A8AEC3060D11FBF3210B650A1CBFB651BAA2A181F9352963710917491A77CD60`. The original 1.5 MB `dist/default.zero` is not duplicated here: the repository already contains byte-identical model data at `project-chromatic/dist/default.zero`, SHA-256 `F6A00902C23E7F32F96B028AC36464CC9012CD8774D5E6B0DEE5E1C94E1C03C9` (Git blob `93aa9788b3ed9e91e14f10b9d8171dea7fdef315`). For local build/test layouts that expect a sibling `dist/default.zero`, copy that existing file locally; do not commit a duplicate.

The included teacher dataset is a small 128-position historical example used for provenance/reproduction, not the requested future million-position dataset. No match PGNs, UCI traces, binaries, `.git` metadata, or secrets are included.
