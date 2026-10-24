# Reproduce the memory fix comparison

The source, model and new executable are in the sibling project-chromatic-memory package.
Use its README for direct native probe commands. legacy-depth_probe.exe is the
preserved old diagnostic executable used in these logs; use the same default.zero.
Its complete preserved source is in legacy-source/; original baseline sources
and the GPL-3.0 license are included. To run legacy-source CTest, first copy the
shared default.zero into legacy-source/dist/default.zero. The native probe can
instead use --model with the path to the new package's identical model.

Raw request JSON files contain exact arguments. Adjust only executable/model paths
when replaying a run elsewhere. Raw stdout files are unmodified probe responses.

Plot reproduction: python memory_comparison_report.py --input ../runs.jsonl --out ..
from this directory. Package versions are pinned in requirements.txt.

The full experiment/verification script snapshots expect the original workspace
layout: copy them to work/ to rerun. Both preserved projects, original manifests
and previous matched-budgets.json are intentionally read for integrity and budget
checks. Completed-N comparisons exclude aborted allocation attempts.
