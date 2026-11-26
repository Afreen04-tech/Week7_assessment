"""Small reproducible retrieval demo. Expand the query list to your full 12-question set."""
import time
from retrieval import HybridRetriever
from query_rewriting import rewritten_search

QUESTIONS=[
 ('How many casual leaves per year?','leave_policy.txt:0'),
 ('How many sick leaves are allowed?','leave_policy.txt:1'),
 ('What is the daily work schedule?','work_policy.txt:0'),
 ('How many holidays does the company observe?','holiday_policy.txt:0'),
 ('What is the meal reimbursement cap?','expense_policy.txt:1'),
 ('What is the probation duration?','probation_policy.txt:0'),
 ('Can staff work remotely?','remote_work_policy.txt:0'),
 ('What password length is required?','it_security_policy.txt:1'),
 ('How long is maternity leave?','parental_leave_policy.txt:0'),
 ('When should receipts be submitted?','expense_policy.txt:2'),
 ('How many days per week are employees expected to work?','work_policy.txt:2'),
 ('Who should receive phishing reports?','it_security_policy.txt:2'),
]
def hit3(results, expected): return any(x['id']==expected for x in results[:3])

def main():
    r=HybridRetriever(); methods={'BM25':r.bm25_search,'Vector':r.vector_search,'Hybrid':r.hybrid_search}
    for name,fn in methods.items():
        hits=sum(hit3(fn(q,3),expected) for q,expected in QUESTIONS)
        print(f'{name} hit@3: {hits}/{len(QUESTIONS)} = {hits/len(QUESTIONS):.2f}')
    start=time.perf_counter(); hits=sum(hit3(rewritten_search(r,q,3),e) for q,e in QUESTIONS); elapsed=time.perf_counter()-start
    print(f'Hybrid + query rewriting hit@3: {hits}/{len(QUESTIONS)}; latency={elapsed:.3f}s')
    print('Metadata filter example:', [x['id'] for x in r.filtered_search('leave policy',department='leave',year=2026)])

if __name__=='__main__': main()
