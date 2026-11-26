"""Citation formatting, extraction and validation helpers."""
import re
CITATION_RE=re.compile(r"\[Source:\s*([^,\]]+),\s*Page\s*(\d+)\]",re.IGNORECASE)

def format_chunk(text, source, page): return f"[Source: {source}, Page {page}] {text}"
def extract_citations(answer): return [(m.group(1).strip(),int(m.group(2))) for m in CITATION_RE.finditer(answer)]
def validate_citations(answer, retrieved_chunks):
    allowed={(c['metadata']['source'],int(c['metadata'].get('page',1))) for c in retrieved_chunks}
    found=extract_citations(answer)
    return {"citations":found,"invalid":[c for c in found if c not in allowed],"valid":bool(found) and all(c in allowed for c in found)}
