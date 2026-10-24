"""Build a bounded, deterministic mixed-head validation batch from a fixed set.

This tool reads JSONL only. It performs no training, GPU operation or download.
"""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import argparse
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source,output,maximum=1000):
    source=Path(source).resolve();output=Path(output).resolve()
    rows=[]
    with source.open('rb') as stream:
        for line,raw in enumerate(stream,1):
            if not raw.strip():continue
            if len(rows)>=maximum:raise ValueError('Fixed numeric source exceeds its explicit bounded record limit')
            record=json.loads(raw.decode('utf-8'));rows.append((line,raw,record))
    weak=[r for r in rows if float(r[2].get('value_weight',0))==0 and r[2].get('source')=='lichess-openings-cc0']
    positive={phase:[r for r in rows if float(r[2].get('value_weight',0))>0 and r[2].get('phase')==phase]
              for phase in ['opening','middlegame','endgame']}
    if len(weak)<16 or any(len(positive[p])<n for p,n in [('opening',6),('middlegame',5),('endgame',5)]):
        raise ValueError('Need 16 weak opening anchors and positive-value rows in all 3 phases (6/5/5)')
    selected=list(weak[:16]);cursor={phase:0 for phase in positive}
    for index in range(16):
        phase=['opening','middlegame','endgame'][index%3];selected.append(positive[phase][cursor[phase]]);cursor[phase]+=1
    identities=[item[2]['identity'] for item in selected]
    if len(set(identities))!=32:raise ValueError('Selected diagnostic positions must have unique identities')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('wb') as stream:
        for _,raw,_ in selected:stream.write(raw if raw.endswith(b'\n') else raw+b'\n')
    spread_indices=[round(index*31/9) for index in range(10)]
    native=output.with_name('numeric-native-inputs.jsonl')
    with native.open('wb') as stream:
        for index in spread_indices:
            raw=selected[index][1];stream.write(raw if raw.endswith(b'\n') else raw+b'\n')
    metadata={'format':'chromatic-arc-numeric-inputs-v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'source_file':str(source),'source_sha256':digest(source),'source_rows':len(rows),'source_split':'fixed test',
        'output_file':str(output),'output_sha256':digest(output),'rows':32,'weak_anchor_rows':16,'positive_value_rows':16,
        'positive_phase_counts':dict(Counter(item[2]['phase'] for item in selected[16:])),
        'policy_weight_sum':sum(float(item[2]['policy_weight']) for item in selected),
        'value_weight_sum':sum(float(item[2]['value_weight']) for item in selected),
        'selection':'first 16 weak anchors in frozen file order; positive rows round robin opening/middlegame/endgame with frozen order inside each phase',
        'scope':'Numerical validation only; these rows are never added to production training or saved as a trained candidate.',
        'records':[{'index':index,'source_line':line,'identity':record['identity'],'phase':record['phase'],
            'policy_weight':record['policy_weight'],'value_weight':record['value_weight'],'source':record.get('source'),
            'raw_line_sha256':hashlib.sha256(raw).hexdigest()} for index,(line,raw,record) in enumerate(selected)],
        'native_inputs':{'file':str(native),'sha256':digest(native),'rows':10,'selected_batch_indices':spread_indices,
            'phase_counts':dict(Counter(selected[index][2]['phase'] for index in spread_indices)),
            'positive_value_rows':sum(float(selected[index][2]['value_weight'])>0 for index in spread_indices)}}
    meta=output.with_suffix('.meta.json');meta.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return metadata


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,default=ROOT/'reports/numeric-inputs.jsonl')
    a=p.parse_args(argv);report=prepare(a.source,a.output)
    print(json.dumps({key:report[key] for key in ['rows','weak_anchor_rows','positive_value_rows','positive_phase_counts','output_sha256']},ensure_ascii=False));return 0


if __name__=='__main__':raise SystemExit(main())
