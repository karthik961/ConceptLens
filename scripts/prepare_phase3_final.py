"""Freeze human-reviewed, project-authored Phase 3 benchmark using Mac Chroma.

All gold labels are written only to a LOCAL file. Only model-input contexts go to Hopper.
No candidate evaluation results should be used to revise the frozen benchmark.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

REQUIRED = ('case_id', 'subject', 'topic', 'question', 'student_answer', 'expected_assessment', 'teacher_rationale', 'reference_source')
LABELS = ('correct', 'incomplete', 'incorrect', 'off_topic')
MODEL_FILES = ('src/feedback_engine.py', 'src/feedback_schema.py', 'src/assessment_checklist.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_review(path):
    with Path(path).open('r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 96:
        raise ValueError(f'Expected 96 cases; found {len(rows)}')
    ids = set(); questions = set(); counts = collections.Counter()
    for r in rows:
        if not all(r.get(k, '').strip() for k in REQUIRED):
            raise ValueError('Missing required cell: ' + repr(r))
        if r['case_id'] in ids or r['question'].casefold().strip() in questions:
            raise ValueError('Repeated case ID or question: ' + r['case_id'])
        ids.add(r['case_id']); questions.add(r['question'].casefold().strip())
        if r['expected_assessment'] not in LABELS:
            raise ValueError('Unknown label for '+r['case_id'])
        if r['subject'] not in ('computer_science','electronics'):
            raise ValueError('Unknown subject for '+r['case_id'])
        if not r['reference_source'].startswith(r['subject']+'/'):
            raise ValueError('Reference/source mismatch for '+r['case_id'])
        counts[(r['subject'],r['expected_assessment'])] += 1
    for subject in ('computer_science','electronics'):
        for label in LABELS:
            if counts[(subject,label)] != 12:
                raise ValueError(f'Unbalanced {subject}/{label}: {counts[(subject,label)]} vs 12')
    return sorted(rows,key=lambda r:r['case_id'])


def freeze_contexts(rows, store, retriever):
    prepared=[]
    for row in rows:
        retrieval=retriever(store,row['question'],row['subject'],k=4)
        hits=retrieval['hits']
        if len(hits) != 4:
            raise ValueError(f"{row['case_id']}: wanted 4 distinct retrieved sources; got {len(hits)}")
        sources=[]
        for h in hits:
            doc=h['document']
            if not doc.page_content.strip():raise ValueError('Empty retrieved passage')
            sources.append({'text':doc.page_content,'metadata':dict(doc.metadata), 'distance':float(h['distance'])})
        if len({x['metadata']['source'] for x in sources})!=4:
            raise ValueError('Duplicate sources for '+row['case_id'])
        # Deliberately exclude expected_assessment, rationale, topic, and reference_source.
        prepared.append({'case_id':row['case_id'], 'subject':row['subject'],
                         'question':row['question'], 'answer':row['student_answer'],
                         'k':4,'sources':sources})
    return prepared


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--review',default='eval/phase3_final_review.csv')
    parser.add_argument('--approved',action='store_true',help='I personally reviewed the answer labels/rationales and approve freezing them')
    parser.add_argument('--out-prefix',default='eval/phase3_final')
    args=parser.parse_args()
    if not args.approved:
        parser.error('Review all 96 examples first; run again with --approved only after reviewing')
    from src.settings import CHROMA_DIR, EMBED_MODEL
    from src.knowledge import create_store
    from src.retrieval import unique_sources_search
    from src import knowledge
    if CHROMA_DIR.name != '.chromadb_1100':
        parser.error(f'Expected .chromadb_1100, got {CHROMA_DIR}')
    if not CHROMA_DIR.is_dir():
        parser.error('Chroma index missing; run python -m scripts.index_knowledge first')
    import inspect
    if 'chunk_size=1100, chunk_overlap=100' not in inspect.getsource(knowledge.index_documents):
        parser.error('Expected 1100-character chunking code')
    base=Path(args.out_prefix)
    gold_path=base.with_name(base.name+'_gold_LOCAL_ONLY.csv')
    contexts_path=base.with_name(base.name+'_contexts.jsonl')
    manifest_path=base.with_name(base.name+'_manifest.json')
    for p in (gold_path, contexts_path, manifest_path):
        if p.exists():parser.error(f'Refusing to overwrite already frozen file: {p}')
    rows=read_review(args.review)
    store=create_store()
    if len(store.get()['ids']) != 32:
        parser.error('Expected exactly 32 indexed chunks, with no user-upload chunks')
    contexts=freeze_contexts(rows,store,unique_sources_search)
    gold_path.parent.mkdir(parents=True,exist_ok=True)
    # Write contexts first, gold locally only. Gold never belongs in Hopper tar.
    with contexts_path.open('x',encoding='utf-8') as f:
        for r in contexts:f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')
    with gold_path.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['case_id','subject','expected_assessment','teacher_rationale','reference_source'])
        w.writeheader()
        for r in rows:w.writerow({key:r[key] for key in w.fieldnames})
    corpus_files=[Path('knowledge/catalog.json'),*sorted(Path('knowledge/computer_science').glob('*.md')),*sorted(Path('knowledge/electronics').glob('*.md'))]
    manifest={
      'protocol':'ConceptLens Phase3 final, project-authored examples reviewed by project author; no tuning on this set',
      'cases':96,'distribution':{'subjects':{'computer_science':48,'electronics':48},'each_label_per_subject':12},
      'model':'qwen2.5:7b','prompt_variant':'checklist','retrieval_k':4,
      'chunk_size':1100,'chunk_overlap':100,'embedding_model':EMBED_MODEL,
      'index_chunks':32,'review_file_sha256':digest(args.review),
      'contexts_sha256':digest(contexts_path),'gold_sha256':digest(gold_path),
      'model_code_sha256':{p:digest(p) for p in MODEL_FILES},
      'knowledge_sha256':{str(p):digest(p) for p in corpus_files if p.is_file()}
    }
    with manifest_path.open('x',encoding='utf-8') as f:json.dump(manifest,f,indent=2,sort_keys=True)
    print('FROZEN 96 cases: 12 for each subject × label combination')
    print('Hopper-only inputs:',contexts_path,manifest_path)
    print('KEEP ON MAC:',gold_path,args.review)
    print('Contexts SHA256:',manifest['contexts_sha256'])
    print('Gold SHA256:',manifest['gold_sha256'])
    print('Do NOT revise prompts, code, labels, or cases after looking at evaluation predictions.')
if __name__=='__main__':main()
