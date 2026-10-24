import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zero.data import atomic_json
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--executable',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();path=Path(args.executable).resolve()
    result=subprocess.run([str(path)],cwd=path.parent,capture_output=True,text=True,timeout=30)
    caught=result.returncode!=0 and 'WDL child-to-parent sign inversion' in result.stderr
    report={'mutation':'remove the single child-to-parent value sign inversion',
            'caught_by_regression':caught,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
    atomic_json(args.output,report);print(json.dumps(report,indent=2))
    if not caught:raise SystemExit('sign mutation escaped regression')
