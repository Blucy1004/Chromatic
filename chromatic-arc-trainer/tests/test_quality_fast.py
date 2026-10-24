from copy import deepcopy
from pathlib import Path
import json
import pytest
import chess
from shardtrain import quality as base
from shardtrain.calibration import Calibration
from shardtrain.quality_fast import normalize_selected,semantic_record

FEN=' '.join(chess.STARTING_FEN.split()[:4])
def entry(depth=24,knodes=100,cp=20,line='e2e4 e7e5',pvs=None):
    return {'depth':depth,'knodes':knodes,'pvs':pvs if pvs is not None else [{'cp':cp,'line':line}]}
def record(entries=None,fen=FEN):return {'fen':fen,'evals':entries if entries is not None else [entry()]}
def compare(raw,config=None,calibration=None):
    source={'license':'CC0-1.0','unit_test_fixture':True,'line':3}
    old=base.parse_record(raw,calibration=calibration,config=config,source=source)
    new=normalize_selected(raw,config=config,calibration=calibration,source=source)
    assert semantic_record(old)==semantic_record(new)
    return new

@pytest.mark.parametrize('raw',[
    record(),record([entry(depth=18),entry(depth=30)]),
    record([entry(knodes=20),entry(knodes=1000)]),
    record([entry(),entry(pvs=[{'cp':10,'line':'d2d4'},{'cp':20,'line':'e2e4'}])]),
    record([entry(depth=30,line='e2e4 e7e4'),entry(depth=18,line='d2d4 d7d5')]),
    record([entry(pvs=[{'cp':20,'line':'e2e4'},{'cp':10,'line':'e2e4'}]),entry(depth=18)]),
    record([entry(),entry(cp=90,line='d2d4 d7d5')]),
    record([entry(depth=18,line='e2e4 e7e4'),entry(depth=30)]),
    record([entry(depth=2),entry(knodes=1),entry(depth=30)]),
    record([entry(pvs=[{'cp':100,'line':'e7e5'},{'cp':-50,'line':'d7d5'}])],FEN.replace(' w ',' b ')),
    record([entry(pvs=[{'mate':-3,'line':'e7e5'}])],FEN.replace(' w ',' b ')),
    record([entry(pvs=[{'cp':30,'line':'e1h1 e8a8'}])],'r3k2r/8/8/8/8/8/8/R3K2R w KQkq -'),
    record([entry(pvs=[{'cp':0,'line':'a7a8n'},{'cp':50,'line':'a7a8q'}])],'4k3/P7/8/8/8/8/8/4K3 w - -'),
])
def test_selected_tensor_semantics_match_full(raw):compare(raw)

def test_tie_keeps_original_analysis_order():
    selected=compare(record([entry(cp=10),entry(cp=200,line='d2d4 d7d5')]))
    assert selected['source_eval']['selected_evaluation_index']==0
    assert selected['source_eval']['cp_white']==10

def test_skipped_lower_pvs_are_not_validated_or_claimed_valid(monkeypatch):
    calls=[];original=base._pv
    def spy(board,pv,config):calls.append(pv['line']);return original(board,pv,config)
    monkeypatch.setattr(base,'_pv',spy)
    raw=record([entry(depth=18,line='e2e4 e7e4'),entry(depth=30)])
    selected=normalize_selected(raw)
    assert calls==['e2e4 e7e5']
    assert selected['source_eval']['skipped_lower_evaluation_indices']==[0]
    assert selected['source_eval']['rejected_evaluations']==[]
    assert 'uninspected' in selected['source_eval']['rejected_evaluations_scope']

@pytest.mark.parametrize('raw',[
    None,{'fen':FEN,'evals':[]},record([entry(depth=True)]),record([entry(knodes=float('inf'))]),
    record([entry(depth=2,knodes=None)]),record([entry(depth=24.5)]),
    record([entry(pvs=[])]),record([entry(pvs='bad')]),record([entry(pvs=[{'cp':float('nan'),'line':'e2e4'}])]),
    record([entry(depth=2),entry(depth=30,line='e2e4 e7e4')]),
    record([entry()],fen='8/8/8/8/8/8/8/8 w - -'),record([entry(depth=30)],fen='bad'),
])
def test_rejection_reason_matches_original_order(raw):
    with pytest.raises(base.QualityError) as old:base.parse_record(raw)
    with pytest.raises(base.QualityError) as new:normalize_selected(raw)
    assert new.value.reason==old.value.reason

def test_metadata_and_full_failure_audit_is_partial_and_ordered():
    raw=record([entry(depth=2),entry(depth=30,line='e2e4 e7e4'),entry(depth=18)])
    selected=compare(raw);audit=selected['source_eval']
    assert audit['selected_evaluation_index']==2
    assert audit['rejected_evaluations']==['low_depth','illegal_pv']
    assert audit['full_validation_attempted_indices']==[1,2]

def test_calibrated_labels_and_full_selected_pv_remain_identical():
    root=Path(__file__).resolve().parents[1]
    calibration=Calibration.load(root/'fixtures/calibration.json')
    selected=compare(record([entry(depth=18),entry(depth=30,line='e2e4 e7e5 g1f3 b8c6')]),calibration=calibration)
    assert selected['source_eval']['pv_plies_checked']==[4]
    assert selected['value_weight']>0

def test_invalid_configuration_and_large_integer_keep_base_priority():
    cfg=base.QualityConfig(min_depth=0)
    with pytest.raises(ValueError) as old:base.parse_record(record(),config=cfg)
    with pytest.raises(ValueError) as new:normalize_selected(record(),config=cfg)
    assert str(old.value)==str(new.value)
    # Metadata conversion overflow must not hide the earlier invalid-FEN error.
    raw=record([entry(depth=10**1000)],fen='bad')
    with pytest.raises(base.QualityError) as old:base.parse_record(raw)
    with pytest.raises(base.QualityError) as new:normalize_selected(raw)
    assert old.value.reason==new.value.reason

def test_bounded_corpus_comparison_compares_every_semantic_field():
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'tools/validate_quality_fast.py'
    spec=importlib.util.spec_from_file_location('arc_validate_quality_fast_unit',path)
    tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)
    rows=[(1,record([entry(depth=18,line='e2e4 e7e4'),entry(depth=30)])),
          (2,record([entry(depth=2),entry(depth=30,line='e2e4 e7e4')])),(3,record())]
    result=tool.compare_records(rows,max_records=3,deadline_seconds=10)
    assert result['semantic_equivalent'] and result['mismatch_count']==0
    assert result['records']==3 and result['accepted']=={'full':2,'selected':2}
    assert result['selected_audit_totals']['lower_analyses_skipped_uninspected']==1
    # The all-invalid record deliberately replays the full parser to preserve
    # its original rejection priority, so this tiny mixed sample ties PV calls.
    assert result['calls']['selected']['_pv']<=result['calls']['full']['_pv']

def test_unused_huge_score_keeps_historical_record_level_overflow():
    raw=record([entry(depth=30),entry(depth=18,cp=10**1000)])
    with pytest.raises(OverflowError):base.parse_record(raw)
    with pytest.raises(OverflowError):normalize_selected(raw)
    # If an earlier PV in that same analysis is already illegal, the full
    # parser never reaches its huge later score; fallback must preserve success.
    raw=record([entry(depth=30),entry(depth=18,pvs=[{'cp':20,'line':'e2e5'},{'cp':10**1000,'line':'d2d4'}])])
    compare(raw)
