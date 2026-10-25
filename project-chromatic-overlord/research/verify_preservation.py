#!/usr/bin/env python3
"""Recheck historical byte identities; this script never rewrites protected paths."""
import argparse,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(p):
    digest=hashlib.sha256()
    with p.open('rb') as f:
        while data:=f.read(1024*1024):digest.update(data)
    return digest.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--base',default='ae222e7');a=p.parse_args()
    initial=json.loads((ROOT/'project-chromatic-overlord/reports/preserved-sha256.json').read_text())
    provenance=json.loads((ROOT/'project-chromatic-overlord/reports/provenance/models.json').read_text())['protected_artifact_sha256']
    result={'base':a.base,'protected_groups':[],'previous_reports_changed':[],'passed':False}
    for label,paths in [('original-source-deployments',initial),('all-preserved-models-and-binaries',provenance)]:
        checked=[]
        for path,expected in paths.items():
            file=Path(path) if Path(path).is_absolute() else ROOT/path
            actual=sha(file);assert actual==expected,(str(file),expected,actual);checked.append({'path':str(file),'sha256':actual})
        result['protected_groups'].append({'name':label,'count':len(checked),'files':checked})
    prefix='project-chromatic-overlord/reports'
    previous=set(subprocess.check_output(['git','ls-tree','-r','--name-only',a.base,'--',prefix],cwd=ROOT,text=True).splitlines())
    changed=set(subprocess.check_output(['git','diff','--name-only',a.base,'--',prefix],cwd=ROOT,text=True).splitlines())
    result['previous_reports_changed']=sorted(previous&changed);assert not result['previous_reports_changed']
    result['previous_report_files']=len(previous);result['passed']=True
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'protected_files':[g['count'] for g in result['protected_groups']],'previous_reports':len(previous)}))
if __name__=='__main__':main()
