from __future__ import annotations

# Allow both `python -m part3.<script>` and direct `python part3/<script>.py`.
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from part3.rag import LocalRetriever

ROOT=Path(__file__).resolve().parents[1]
CASES=[
 {'query':'What is the footwear return period?','relevant':['KB002']},
 {'query':'How long can COD refunds take?','relevant':['KB005']},
 {'query':'What is the electronics return window?','relevant':['KB003']},
 {'query':'How long is standard delivery?','relevant':['KB007']},
 {'query':'When is reverse pickup available?','relevant':['KB009']},
]

def main():
 r=LocalRetriever(); rows=[]
 for c in CASES:
  hits=r.retrieve(c['query'],k=3); docs=[]
  for h in hits:
   if h['doc_id'] not in docs: docs.append(h['doc_id'])
  rel=set(c['relevant']); ret=set(docs[:3])
  p=len(rel & ret)/3; rec=len(rel & ret)/len(rel)
  rows.append({'query':c['query'],'relevant_documents':sorted(rel),'retrieved_documents':docs[:3],'precision_at_3':p,'recall_at_3':rec})
 print(json.dumps(rows,indent=2)); (ROOT/'results'/'retrieval_evaluation.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
if __name__=='__main__': main()
