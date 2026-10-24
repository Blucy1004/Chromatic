from pathlib import Path
import importlib.util
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('preserved_depth_verification',ROOT/'work/verify_depth_delivery.py')
verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
verifier.PROJECT=ROOT/'outputs/project-chromatic-memory'
verifier.OUT=ROOT/'outputs/chromatic-memory-analysis'
verifier.RAW=verifier.OUT/'validation-uci-raw'
verifier.MODEL=verifier.PROJECT/'dist/default.zero'
verifier.NEW_EXE=verifier.PROJECT/'build/chromatic_depth.exe'

def check_memory_telemetry(t,simulations,result=None):
    check=verifier.check
    check(t['schema_version']==2 and not t['proof_only'],'memory telemetry schema/proof scope')
    h=t['leaf_depth'];c=t['counters']
    check(sum(h)==simulations and h[0]==0,'one endpoint per completed simulation')
    verifier.check_summary(t['depth'],h)
    tail=0
    for depth in range(len(h)-1,-1,-1):
        tail+=h[depth]
        check(t['traversal_depth'][depth]==tail,'traversal tail sum')
        check(t['pre_failure_leaf_depth'][depth]+t['post_failure_leaf_depth'][depth]==h[depth],'phase partition')
        check(sum(root['leaf_depth'][depth] for root in t['roots'])==h[depth],'root histogram partition')
    for root in t['roots']:check(sum(root['leaf_depth'])==root['visits'],'root histogram visits')
    check(sum(root['visits'] for root in t['roots'])==simulations,'root visits completed N')
    check(c['attempted_simulations']-c['aborted_simulations']==simulations,'aborted attempts are not completed visits')
    check(c['non_retained_neural_evaluations']==0 and c['discarded_neural_evaluations']==0,'no failed-attempt NN values')
    failures=sum(c[k] for k in ['child_fail_node_limit','child_fail_byte_capacity','expansion_fail_byte_capacity'])
    event=t['first_allocation_failure']
    check(event['occurred']==bool(failures),'failure event/counters')
    if event['occurred']:
        check(event['simulation']==event['completed_simulations']+1,'first failure next-completed index')
        check(sum(t['pre_failure_leaf_depth'])==event['simulation']-1,'pre-failure completed count')
    else:check(t['pre_failure_leaf_depth']==h and sum(t['post_failure_leaf_depth'])==0,'no failure phase')
    previous=[0]*len(h);previous_ms=0
    for snap in t['snapshots']:
        verifier.check_summary(snap['depth'],snap['leaf_depth'])
        verifier.check_summary(snap['window_depth'],snap['window_leaf_depth'])
        check(snap['simulation']==sum(snap['leaf_depth']),'snapshot completed N')
        check(snap['elapsed_ms']>=previous_ms,'snapshot time ordering')
        check(all(a+b==z for a,b,z in zip(previous,snap['window_leaf_depth'],snap['leaf_depth'])),'snapshot window partition')
        check(sum(root['visits'] for root in snap['roots'])==snap['simulation'],'snapshot root visits')
        previous=snap['leaf_depth'];previous_ms=snap['elapsed_ms']
    check(previous==h,'final snapshot complete')
    for p in t['prunes']:
        check(p['root_statistics_preserved'] and p['roots_before']==p['roots_after'],'root statistics preserved across compaction')
    if result is not None:
        check(c['node_creations']+1-c['nodes_evicted']==result['arena_nodes'],'cumulative creations minus evictions')
        check(c['neural_evaluations']==result['neural_evals'],'NN count')
        check(t['depth']['max_depth']==result['max_depth'],'max depth')
        lookup={r['uci']:r for r in result['visits']}
        for root in t['roots']:
            other=lookup[root['uci']]
            check(root['visits']==other['n'] and root['q']==other['q'] and root['prior']==other['prior'],'root serialization')
    return {'samples':simulations,**{k:t['depth'][k] for k in ['max_depth','mean_depth','median_depth']},
            'snapshots':len(t['snapshots']),'prunes':c['prune_count'],'all_histogram_and_counter_checks_passed':True}

verifier.check_telemetry=check_memory_telemetry

if __name__=='__main__':
    # All four comparisons stop before the first memory failure, so existing
    # release-equivalence checks apply unchanged to this prefix.
    verifier.main()
    with verifier.LoggedNative(verifier.NEW_EXE,'new-post-saturation-uci') as engine:
        verifier.configure(engine,True)
        result=verifier.uci_search(engine,verifier.chess.Board(),'go nodes 131072',True)
        engine.ready()
    path=verifier.OUT/'validation-uci.json'
    data=json.loads(path.read_text(encoding='utf-8'))
    data['post_saturation_go_nodes']=result
    data['scope_note']='Production parity is pre-saturation; post-saturation divergence is the intended bounded-tree change.'
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    assert result['info']['nodes']==131072 and result['telemetry_received']
    print('MEMORY_UCI_VALIDATION_PASSED')
