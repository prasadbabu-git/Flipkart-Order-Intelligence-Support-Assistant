from __future__ import annotations
import re
INJECTION_PATTERNS=[r'ignore\s+(all|any|previous)\s+instructions',r'ignore\s+previous\s+instructions',r'pretend\s+you\s+are',r'disregard\s+the\s+rules']

def check_input(text:str):
    low=text.lower()
    matched=[p for p in INJECTION_PATTERNS if re.search(p,low)]
    return {'blocked':bool(matched),'patterns':matched}
