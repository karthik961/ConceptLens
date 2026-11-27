"""Run one third of a frozen 96-case FINAL benchmark on Hopper GPU.

No gold/expected labels are read by this file or sent to the model.
"""
from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
from types import SimpleNamespace


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_manifest(manifest_file,context_file):
    m=json.loads(Path(manifest_file).read_text(encoding='utf-8'))
    if m.get('cases')!=96 or m.get('model')!='qwen2.5:7b' or m.get('prompt_variant')!='checklist' or m.get('retrieval_k')!=4:
        raise ValueError('Unexpected model, prompt, k, or count; do not run')
    if digest(context_file)!=m['contexts_sha256']:
        raise ValueError('Frozen context file has changed')
    for p,expected in m['model_code_sha256'].items():
        if digest(p)!=expected:
            raise ValueError(f'Code differs from Mac freeze: {p} (sync exact frozen files first)')
    return m


def load_contexts(path):
    cases=[json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(cases)!=96 or len({c['case_id'] for c in cases})!=96:
        raise ValueError('Not exactly 96 distinct benchmark cases')
    for c in cases:
        if any(k in c for k in ('expected_assessment','teacher_rationale','reference_source','gold_label','topic')):
            raise ValueError('Gold/teacher data contaminated model context')
        if c['k']!=4 or len(c['sources'])!=4:
            raise ValueError('Incorrect frozen retrieval context')
    return sorted(cases,key=lambda c:c['case_id'])


def fixed_retriever(row):
    def run(_store,question,subject,k=4):
        if (question,subject,k)!=(row['question'],row['subject'],4):
            raise ValueError('Retriever input mismatch')
        hits=[]
        for source in row['sources']:
            doc=SimpleNamespace(page_content=source['text'],metadata=source['metadata'])
            hits.append({'document':doc,'source':source['metadata']['source'],'distance':float(source['distance'])})
        return {'hits':hits,'latency_ms':None}
    return run


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--contexts',default='eval/phase3_final_contexts.jsonl')
    a.add_argument('--manifest',default='eval/phase3_final_manifest.json')
    a.add_argument('--part',type=int,choices=[0,1,2],required=True)
    a.add_argument('--output',required=True)
    args=a.parse_args()
    m=load_manifest(args.manifest,args.contexts)
    cases=load_contexts(args.contexts)
    selected=cases[args.part*32:(args.part+1)*32]
    assert len(selected)==32
    # Model loaded only after integrity preflight.
    from langchain_ollama import ChatOllama
    from src.feedback_engine import give_feedback
    model=ChatOllama(model='qwen2.5:7b',temperature=0,num_ctx=4096,
       base_url='http://'+os.environ.get('OLLAMA_HOST','127.0.0.1:11434'))
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():a.error(f'Output already exists; never overwrite experimental results: {out}')
    errors=0
    with out.open('x',encoding='utf-8') as f:
        for i,case in enumerate(selected,1):
            t=time.perf_counter()
            r={'case_id':case['case_id'],'subject':case['subject'],'question':case['question'],
               'answer':case['answer'],'part':args.part,'model':'qwen2.5:7b','prompt_variant':'checklist',
               'retrieval_k':4,'contexts_sha256':m['contexts_sha256'],
               'retrieved_sources':[s['metadata']['source'] for s in case['sources']]}
            try:
                generated=give_feedback(object(),case['subject'],case['question'],case['answer'],
                     k=4,mode='rag',model=model,retriever=fixed_retriever(case),prompt_variant='checklist')
                r['result']=generated
                r['assessment']=generated['assessment']
            except Exception as ex:
                r['error']=type(ex).__name__+': '+str(ex)
                errors+=1
            r['generation_ms']=round((time.perf_counter()-t)*1000,1)
            f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
            print(f"{args.part}:{i:02d}/32 {case['case_id']} {r.get('assessment',r.get('error'))} {r['generation_ms']/1000:.1f}s",flush=True)
    print('Saved',out,'records',len(selected),'errors',errors,flush=True)
    if errors:raise SystemExit(1)
if __name__=='__main__':main()
