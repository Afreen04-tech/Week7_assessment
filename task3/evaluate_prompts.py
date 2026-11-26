"""Prompt comparison harness; use generated answers to label hallucinations manually."""
PROMPTS={
'basic':'Answer the question.',
'context_only':'Answer ONLY using the context. If absent, say you do not know.',
'grounded_cited':'Answer ONLY using the context. Cite each factual claim as [Source: filename, Page n]. If absent, say exactly I do not know.'
}
QUESTIONS=['How many casual leaves per year?','How many sick leaves?','What are working hours?','How many public holidays?','What is the meal expense cap?','What is probation length?','What is maternity leave?','What is the password requirement?','Who won the world cup?','What is the CEO name?']
if __name__=='__main__':
    for name,prompt in PROMPTS.items(): print(f'{name}: {prompt}')
    print(f'Questions prepared: {len(QUESTIONS)} (3 unanswerable)')
    print('Record hallucination counts after running each prompt against the same retrieved contexts.')
