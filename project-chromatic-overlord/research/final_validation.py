#!/usr/bin/env python3
"""Collect current native evidence and fail on missing/partial/mismatched checks."""
import argparse,collections,csv,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--linux',type=Path,required=True);p.add_argument('--windows',type=Path,required=True);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result={'checks':[],'passed':False,'scope':'factorial source and measured executables; old reports preserved'}
    def check(name,passed,**data):result['checks'].append({'name':name,'passed':bool(passed),**data})
    def ctest(name,path):
        data=path.read_text(encoding='utf-8-sig') if path.exists() else ''
        check(name,'100% tests passed, 0 tests failed out of 11' in data,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None)
    expected={'scalar_fixed':16,'screen':128,'legacy_proof':4,'anchors':16,'one_second':32,'five_seconds':8,'ten_seconds':6,'fixed_nodes':128,'fixed_nn':128,'holdout':24}
    manifests=[]
    for name,run in [('linux',a.linux),('windows',a.windows)]:
        manifest=read(run/'manifest.json');manifests.append(manifest);done=read(run/'completed.json') if (run/'completed.json').exists() else {}
        counts=collections.Counter();points=0;attested=True
        for path in run.glob('*/*.meta.json'):
            meta=read(path)
            if meta.get('complete'):counts[path.parent.name]+=1;points+=len(meta['ids']);attested &= meta['loaded_model']==meta['loaded_model_after'] and meta['loaded_model']['sha256']==meta['model_sha256']
        check(name+'_complete_matrix',dict(counts)==expected and points==27048 and attested and done.get('completed') and done['active_seconds']<=5400,
              counts=dict(counts),expected=expected,searches=points,completed=done,model_attestations=attested)
        check(name+'_native_os',manifest['os']==name.capitalize())
    lm,wm=manifests;check('same_cpp_runner_seed_corpora',all(lm[k]==wm[k] for k in ['cpp_sha256','runner_sha256','seed','old_fixture_sha256','holdout_sha256']))
    ctest('linux_release',ROOT/'reports/factorial/preflight/native-linux.log');ctest('windows_msvc_release',a.windows/'ctest.log')
    for name,path in [('linux',ROOT/'reports/factorial/safety/linux-protocol'),('windows',a.artifact)]:
        protocol=read(path/('platform/protocol.json'));research=read(path/('research.json' if name=='linux' else 'research-protocol.json'));manifest=lm if name=='linux' else wm
        check(name+'_uci_perft_stop_quit_time_reclamation',protocol['passed'] and len(protocol['checks'])==41 and protocol['engine_sha256']==manifest['engine_sha256'],checks=len(protocol['checks']),engine_sha256=protocol['engine_sha256'])
        check(name+'_research_protocol',research['passed'] and len(research['checks'])==68 and research['engine_sha256']==manifest['engine_sha256'],checks=len(research['checks']))
    safety=ROOT/'reports/factorial/safety/linux-sanitizers-v2';s=read(safety/'safety.json')
    check('linux_sanitizers_same_cpp',s['passed'] and s['cpp_sha256']==lm['cpp_sha256'],report=str(safety/'safety.json'))
    for kind in ['asan','tsan']:ctest('linux_'+kind,safety/(kind+'-ctest.log'))
    ctest('windows_native_msvc_asan',a.artifact/'asan-ctest.log')
    artifact=read(a.artifact/'artifact.json');check('windows_exact_measured_artifact',artifact['engine_sha256']==wm['engine_sha256'],artifact=artifact)
    preservation=read(ROOT/'reports/factorial/safety/preservation.json');check('historical_preservation',preservation['passed'],protected_counts=[g['count'] for g in preservation['protected_groups']],previous_report_files=preservation['previous_report_files'])
    for name,n in [('statistics',5),('analysis',7)]:
        path=ROOT/f'reports/factorial/safety/unit-{name}.log';data=path.read_text();check('python_'+name,f'Ran {n} tests' in data and data.rstrip().endswith('OK'),path=str(path))
    matches=read(ROOT/'reports/factorial/analysis/matches.json');check('all_game_traces_and_pgns',len(matches)==6 and sum(r['games'] for r in matches)==96 and all(r['pgn_trace_validation'] and not r['promote'] and not r['confirmed_elo_gain'] for r in matches),series=len(matches),games=sum(r['games'] for r in matches))
    source=(ROOT/'include/zero/search.hpp').read_text();uci=(ROOT/'src/main.cpp').read_text()
    check('experimental_defaults_preserved','bool counterexample=false,adaptive_budget=false,forcing_search=false,tactical_verify=false;' in source and 'int threads=1' in source and 'bool research_metrics=false' in source and all(f'{name} type check default false' in uci for name in ['CounterexampleVerify','AdaptiveBudget','ForcingSearch','TacticalVerify','ResearchMetrics']))
    result['known_failures_retained']=[{'path':'reports/factorial/safety/linux-sanitizers/launch-failure.json','kind':'tool PATH launch; no sanitizer test executed; repaired in separate v2'}]
    result['known_failures_retained'] += [{'path':'reports/factorial/analysis-failures/'+name,'kind':'analysis rejected model/OS-specific ten-second diagnostics as unpaired; excluded only those contrasts; primary/workload identity assertions retained'} for name in ['001-unpaired-ten-second-models.log','002-unpaired-ten-second-platforms.log']]
    result['passed']=all(r['passed'] for r in result['checks']);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':result['passed'],'checks':len(result['checks']),'failed':[r['name'] for r in result['checks'] if not r['passed']]}));return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
