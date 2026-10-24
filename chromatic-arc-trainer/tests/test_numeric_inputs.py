"""Mixed input preparation tests use JSON files only, with no torch import."""
from pathlib import Path
import importlib.util
import json
import hashlib

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('prepare_numeric_inputs',ROOT/'tools/prepare_numeric_inputs.py')
prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)


def fixture(path):
    records=[{'identity':f'weak-{i}','source':'lichess-openings-cc0','phase':'opening','policy_weight':.15,'value_weight':0} for i in range(20)]
    records +=[{'identity':f'{phase}-{i}','source':'lichess-evals-cc0','phase':phase,'policy_weight':.25,'value_weight':.25}
        for phase in ['opening','middlegame','endgame'] for i in range(10)]
    path.write_text('\n'.join(json.dumps(row) for row in records)+'\n',encoding='utf-8')
    return path


def test_mixed_selection_has_both_heads_and_phase_diversity(tmp_path):
    source=fixture(tmp_path/'frozen.jsonl');before=source.read_bytes()
    report=prepare.prepare(source,tmp_path/'numeric-inputs.jsonl')
    assert report['weak_anchor_rows']==16 and report['positive_value_rows']==16
    assert report['positive_phase_counts']=={'opening':6,'middlegame':5,'endgame':5}
    assert report['value_weight_sum']>0 and len(set(row['identity'] for row in report['records']))==32
    assert report['source_sha256']==hashlib.sha256(before).hexdigest() and source.read_bytes()==before
    assert report['native_inputs']['rows']==10 and report['native_inputs']['positive_value_rows']==5
    assert set(report['native_inputs']['phase_counts'])=={'opening','middlegame','endgame'}


def test_mixed_selection_is_repeatable_without_changing_payloads(tmp_path):
    source=fixture(tmp_path/'frozen.jsonl')
    first=prepare.prepare(source,tmp_path/'a/numeric-inputs.jsonl')
    second=prepare.prepare(source,tmp_path/'b/numeric-inputs.jsonl')
    assert first['output_sha256']==second['output_sha256']
    original={json.loads(raw)['identity']:raw for raw in source.read_bytes().splitlines()}
    for raw in (tmp_path/'a/numeric-inputs.jsonl').read_bytes().splitlines():assert raw==original[json.loads(raw)['identity']]


def test_missing_positive_value_phase_does_not_make_fake_labels(tmp_path):
    source=fixture(tmp_path/'frozen.jsonl')
    rows=[json.loads(raw) for raw in source.read_text(encoding='utf-8').splitlines()]
    for row in rows:row['value_weight']=0
    source.write_text('\n'.join(json.dumps(row) for row in rows),encoding='utf-8')
    with pytest.raises(ValueError,match='positive-value'):
        prepare.prepare(source,tmp_path/'output.jsonl')
    assert not (tmp_path/'output.jsonl').exists()
