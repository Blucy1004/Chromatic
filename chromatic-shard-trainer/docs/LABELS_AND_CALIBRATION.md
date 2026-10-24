# Label interpretation and independent WDL calibration

The official Lichess Evaluation Database contains CP/mate scores and PVs, not
game results or full WDL targets. Its FENs have four fields: piece placement,
turn, castling and EP. They do not identify game ply, halfmove clock or repetition
history. Default reconstructed clocks are placeholders. The `opening` phase is
a material proxy; explicit first-12-ply coverage comes from separate CC0 opening
anchors with actual move histories.

The official cloud-eval schema specifies both CP and mate from White's point of
view and king-to-rook castling notation. The database exporter passes the cached
scores into its JSONL export. These are primary references:

- https://database.lichess.org/#evals
- https://raw.githubusercontent.com/lichess-org/api/master/doc/specs/schemas/CloudEval.yaml
- https://raw.githubusercontent.com/lichess-org/database/master/src/main/scala/Evals.scala

`quality.py` accepts standard chess only, validates FEN legality, minimum depth
and kilo-node count, and checks every supplied PV move by default. python-chess
normalizes king-to-rook castling before the existing 4672-action encoding is
applied. Underpromotion IDs and absolute-square features match CHROMATIC. Scores
are ranked from the side-to-move perspective; final WDL order is STM win/draw/loss.

The deepest eligible evaluation is selected, with node count as the next key.
Duplicate PV first moves invalidate an evaluation. A lower-depth valid evaluation
can be used if a deeper evaluation contains malformed or illegal data, and that
rejection is recorded. MultiPV labels are softmax over **observed** variations at
100 CP temperature by default. A single PV is weak imitation supervision, never
described as the full optimal policy. Unobserved moves are not certified bad.
Labels carry policy coverage, confidence and explicit independent loss weights.

CP is never divided by a constant and called a probability. Without an accepted
calibration artifact, CP records have `value_weight=0`; their placeholder WDL is
masked out and is not an observed draw. Mate uses its sign separately as weak
teacher WDL; mate zero in a nonterminal position is rejected because its sign
cannot be recovered from JSON integer zero. This does not prove the mate with
the CHROMATIC search or restore unknown repetition/50-move history.

`calibration.py` reads completed official CC0 PGN games with annotated CP and real
Result tags. It excludes early plies by default, limits examples per game and
splits whole game IDs deterministically into fit/validation/test groups. It fits
a symmetric ordinal logistic CP mapping with positive scale and nonnegative draw
margin, chooses parameters by validation NLL and reports untouched-test NLL,
three-class Brier score and reliability ECE. Every game has total weight one.
The test set is never used in fitting or checkpoint selection.

Activation requires enough independent validation and test games, all WDL outcome
classes in the fit data, and validation NLL at least as good as a constant prior.
An unsuccessful artifact remains unaccepted and cannot produce CP value targets.
All calibration position families must be excluded from neural-network training.
Families group identical piece placement even when turn, clocks or castling
rights differ. Anchor-family overlaps should be removed before fitting.

This estimates **empirical game outcomes conditioned on annotated CP**, not
perfect-play WDL. Human strengths, time controls and Stockfish versions vary.
Small calibration samples retain low value weight. Passing software tests or
decreasing training loss is not evidence for model promotion or Elo gain.

The unit tests use explicitly marked synthetic fixtures to verify orientation,
castling, underpromotion, malformed data, group splits and corruption rejection.
Synthetic tests must never be reported as measured calibration performance on
real games. Real experiments should retain the input source URL, version, license,
prefix hash and independent game-group metrics alongside the fitted artifact.
