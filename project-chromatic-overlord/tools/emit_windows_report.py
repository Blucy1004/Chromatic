#!/usr/bin/env python3
import argparse,json
from pathlib import Path

def read(path):
 if path.is_file():return json.loads(path.read_text(encoding='utf-8-sig'))
 return {}
def main():
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--source',required=True);p.add_argument('--run-url',required=True);a=p.parse_args()
 build=read(a.results/'build.json');native=read(a.results/'ctest-status.json');protocol=read(a.results/'platform/protocol.json');matrix=read(a.results/'matrix/completed.json')
 passed=build.get('build_exit')==0 and native.get('exit_code')==0 and protocol.get('passed') is True and protocol.get('native_windows') is True and matrix.get('cells')==18 and matrix.get('os')=='Windows'
 summary={'source_commit':a.source,'run_url':a.run_url,'passed':passed,'native_windows':protocol.get('native_windows',False),'build':build,'ctest':native,'matrix':matrix}
 (a.results/'run-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 doc=Path(__file__).resolve().parents[1]/'WINDOWS_VALIDATION.md';rel=a.results.relative_to(doc.parent).as_posix()
 lines=['# Windows native validation','',f"Status: **{'PASSED' if passed else 'FAILED / INCOMPLETE'}**.",'',f'- Native run: {a.run_url}',f'- Source commit: `{a.source}`.',f'- Runner: Windows x64, MSVC, Visual Studio 2022 CMake generator, Release. Compiler/build details: [{rel}/build.json]({rel}/build.json).',f'- Native CTest exit: {native.get("exit_code","not recorded")}. [Log]({rel}/ctest.log).',f'- Protocol/100ms/1s/10s/model loading/options/recycling/perft: [{rel}/platform/protocol.json]({rel}/platform/protocol.json).',f'- Windows-only 2-model × 3-arm × 3-repeat matrix: [{rel}/matrix/summary.csv]({rel}/matrix/summary.csv).','',
 'Linux results are stored separately under reports/provenance. This document does not substitute a Linux run for native Windows evidence. New search defaults remain off; no champion is promoted.','',
 'The first historical 22/45/56 result recorded a file hash, not a digest from loaded tensors. New runs query `zero model` and compare the canonical in-memory tensor SHA-256 with the exact input file before and after every matrix cell.','',
 '## Windows matrix','', '| Model | Arm | Repeat | Correct / 100 |','|---|---|---:|---:|']
 for row in read(a.results/'matrix/summary.json') or []:lines.append(f"| {row['model']} | {row['arm']} | {row['repeat']} | {row['correct']} |")
 if not passed:lines+=['','Some required evidence is absent or failed. Inspect the retained logs; Windows readiness is not established.']
 doc.write_text('\n'.join(lines)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
