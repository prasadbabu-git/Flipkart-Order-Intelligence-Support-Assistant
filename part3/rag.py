from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
KB_DIR=ROOT/'part3'/'kb'
INDEX_DIR=ROOT/'part3'/'vector_store'
INDEX_DIR.mkdir(exist_ok=True)

class LocalRetriever:
    """Sentence-transformers + FAISS when available; deterministic TF-IDF fallback for offline smoke tests."""
    def __init__(self, kb_dir=KB_DIR):
        self.kb_dir=Path(kb_dir); self.docs=[]; self.chunks=[]; self.backend=None
        for path in sorted(self.kb_dir.glob('KB*.txt')):
            text=path.read_text(encoding='utf-8').strip()
            doc_id=path.stem.split('_')[0]
            self.docs.append({'doc_id':doc_id,'title':path.stem,'text':text})
            for i,s in enumerate([x.strip() for x in re.split(r'(?<=[.!?])\s+', text) if x.strip()],1):
                self.chunks.append({'chunk_id':f'{doc_id}_{i:02d}','doc_id':doc_id,'text':s})
        texts=[c['text'] for c in self.chunks]
        try:
            from sentence_transformers import SentenceTransformer
            self.model=SentenceTransformer('all-MiniLM-L6-v2')
            self.emb=self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            try:
                import faiss
                self.index=faiss.IndexFlatIP(self.emb.shape[1]); self.index.add(np.asarray(self.emb,dtype='float32'))
                self.backend='sentence-transformers+faiss'
            except Exception:
                self.index=None; self.backend='sentence-transformers+numpy'
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.model=TfidfVectorizer(ngram_range=(1,2), norm='l2')
            self.emb=self.model.fit_transform(texts).toarray()
            self.index=None; self.backend='tfidf-fallback'
        (INDEX_DIR/'backend.json').write_text(json.dumps({'backend':self.backend,'chunks':len(self.chunks)},indent=2),encoding='utf-8')

    def retrieve(self, query, k=3):
        if self.backend.startswith('sentence-transformers'):
            q=self.model.encode([query], normalize_embeddings=True)
            if self.index is not None:
                scores,ids=self.index.search(np.asarray(q,dtype='float32'),k)
                pairs=zip(ids[0].tolist(),scores[0].tolist())
            else:
                sims=self.emb @ q[0]; pairs=sorted(enumerate(sims), key=lambda x:x[1], reverse=True)[:k]
        else:
            q=self.model.transform([query]).toarray()[0]
            sims=self.emb @ q; pairs=sorted(enumerate(sims), key=lambda x:x[1], reverse=True)[:k]
        out=[]
        for idx,score in pairs:
            if idx < 0: continue
            c=self.chunks[int(idx)].copy(); c['score']=float(score); out.append(c)
        return out

    def grounded(self, query, threshold=0.15):
        hits=self.retrieve(query,k=3)
        score=hits[0]['score'] if hits else 0.0
        # Require at least one meaningful lexical anchor between the query and
        # the highest-scoring retrieved text. This blocks high-similarity junk
        # matches such as an unrelated policy question about an unseen object.
        stop={'what','is','the','a','an','for','of','to','can','i','how','long','when','does','do','this','that','and','or','on','in','my','please'}
        q_tokens={x for x in re.findall(r'[a-z0-9]+',query.lower()) if len(x)>3 and x not in stop}
        top_text=(hits[0]['text'].lower() if hits else '')
        anchors={x for x in q_tokens if x in top_text}
        grounded=bool(hits) and score>=threshold and bool(anchors)
        return {'grounded':grounded,'score':float(score),'threshold':threshold,'hits':hits,'lexical_anchors':sorted(anchors)}
