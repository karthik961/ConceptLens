"""Score a single frozen final test on Mac. Never tune on this data afterward.

Produces confusion matrix, stratified breakdown, and a citation-audit CSV.
Citation presence is *not* semantic citation support; humans must inspect citations.
"""
from __future__ import annotations
import argparse,csv,glob,hashlib,json,math
from pathlib import Path
from collections import Counter,defaultdict
from scripts.prepare_phase3_final import digest, LABELS

def wilson95(success,total):
    if total==0:return (None,None)
    p=success/total;z=1.95996398454;d=1+z*z/total
    mid=(p+z*z/(2*total))/d
    half=z*math.sqrt((p*(1-p)+z*z/(4*total))/total)/d
    return (max(0,mid-half),min(1,mid+half))

def score(gold_path,manifest_path,results_glob,output_prefix):
    manifest=json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    if digest(gold_path)!=manifest['gold_sha256']:
        raise ValueError('Gold file differs from frozen manifest')
    with Path(gold_path).open(encoding='utf-8',newline='') as f:
        gold={r['case_id']:r for r in csv.DictReader(f)}
    files=sorted(glob.glob(results_glob))
    if not files:raise ValueError('No result files matched')
    preds={}
    for file in files:
        for line in Path(file).read_text(encoding='utf-8').splitlines():
            if not line.strip():continue
            r=json.loads(line);cid=r['case_id']
            if r['contexts_sha256'] != manifest['contexts_sha256']:
                raise ValueError('Mixed runs with different frozen contexts')
            if (r.get('model'),r.get('prompt_variant'),r.get('retrieval_k')) != ('qwen2.5:7b','checklist',4):
                raise ValueError('Unexpected candidate configuration')
            if cid in preds:raise ValueError('Duplicate result for '+cid)
            preds[cid]=r
    if len(preds)!=96 or set(preds)!=set(gold):
        raise ValueError(f'Expected 96 predictions with exact matching IDs; got {len(preds)}; missing {sorted(set(gold)-set(preds))}')
    if any('error' in r for r in preds.values()):
        raise ValueError('At least one inference error; do not silently score incomplete responses')
    stats=defaultdict(lambda:[0,0]); matrix=Counter(); abstentions=0; success=0; flagged=[]; audit=[]
    for cid in sorted(gold):
        g=gold[cid];p=preds[cid];actual=p['assessment'];expected=g['expected_assessment'];correct=actual==expected
        success+=int(correct);abstentions+=int(actual=='cannot_assess')
        for key in ('overall',g['subject'],expected,g['subject']+'/'+expected):
            stats[key][0]+=int(correct);stats[key][1]+=1
        matrix[(expected,actual)]+=1
        result=p['result'];sources=result.get('sources',[]);ids=result.get('evidence_ids',[])
        cited=[]
        for evidence_id in ids:
            if not isinstance(evidence_id,int) or evidence_id<1 or evidence_id>len(sources):
                raise ValueError(f'Invalid citation index in {cid}: {evidence_id}')
            s=sources[evidence_id-1]
            cited.append(f"[{evidence_id}] {s['source']}: {s['text']}")
        flagged.append({'case_id':cid,'subject':g['subject'],'expected':expected,'predicted':actual,
           'correct':int(correct),'question':p['question'],'student_answer':p['answer'],
           'rationale':g['teacher_rationale']})
        audit.append({'case_id':cid,'subject':g['subject'],'expected':expected,'predicted':actual,
          'question':p['question'],'student_answer':p['answer'],
          'explanation':result.get('explanation',''),'misconception':result.get('misconception',''),
          'hint':result.get('hint',''),'citation_count':len(ids),
          'cited_passages':'\n\n'.join(cited),
          'supports_central_claim_0_1_2':'', 'factual_feedback_0_1_2':'', 'reviewer_notes':''})
    prefix=Path(output_prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
    case_path=prefix.with_name(prefix.name+'_case_results.csv')
    audit_path=prefix.with_name(prefix.name+'_citation_audit.csv')
    summary_path=prefix.with_name(prefix.name+'_summary.json')
    for path,rows in ((case_path,flagged),(audit_path,audit)):
        if path.exists():raise ValueError('Output exists; not overwriting: '+str(path))
        with path.open('x',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lo,hi=wilson95(success,96)
    summary={'correct':success,'total':96,'accuracy':success/96,'wilson_95_descriptive':[lo,hi],
       'abstentions':abstentions,'by_group':{k:{'correct':v[0],'total':v[1],'accuracy':v[0]/v[1]} for k,v in stats.items()},
       'confusion':{f'{k[0]}->{k[1]}':v for k,v in matrix.items()},
       'note':'Project-authored, topic-clustered sample: intervals are descriptive, not a guarantee of generalization. Citation support must be human-rated.'}
    with summary_path.open('x',encoding='utf-8') as f:json.dump(summary,f,indent=2,sort_keys=True)
    print(f'FINAL ONE-TIME TEST: {success}/96 = {100*success/96:.1f}% assessment agreement')
    print(f'Wilson interval (descriptive only): {100*lo:.1f}% to {100*hi:.1f}%')
    print('Abstentions:',abstentions)
    for key in sorted(stats):
        if key=='overall':continue
        good,total=stats[key];print(f'{key:35} {good:2d}/{total:2d} = {100*good/total:.1f}%')
    print('Saved:',case_path,summary_path,audit_path)
    print('Review each cited passage against the central claim. Do not treat presence of an ID as evidence support.')
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gold',default='eval/phase3_final_gold_LOCAL_ONLY.csv')
    p.add_argument('--manifest',default='eval/phase3_final_manifest.json')
    p.add_argument('--results',default='eval/phase3_final_7b_*.jsonl')
    p.add_argument('--out-prefix',default='eval/phase3_final_report')
    a=p.parse_args();score(a.gold,a.manifest,a.results,a.out_prefix)
if __name__=='__main__':main()
