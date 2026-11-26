from rag import PolicyRAG, fit_context

ANSWERABLE = [
    "How many casual leaves per year?",
    "How many sick leaves are allowed?",
    "What are the standard working hours?",
    "How many public holidays are there?",
    "What is the daily meal expense limit during approved business travel?",
    "How long is the probation period?",
]
UNANSWERABLE = ["Who won the world cup?", "What is the CEO's name?",
                "What is the company share price?", "Where is the office located?"]

def main():
    rag = PolicyRAG()
    print("ANSWERABLE QUESTIONS")
    for q in ANSWERABLE:
        result = rag.answer(q, k=4)
        print(f"Q: {q}\nA: {result['answer']}\nSources: {result['sources']}\n")
    print("UNANSWERABLE QUESTIONS")
    for q in UNANSWERABLE:
        result = rag.answer(q, k=4)
        print(f"Q: {q}\nA: {result['answer']}\n")
    print("CONTEXT BUDGET TESTS")
    for k in (2, 5, 10):
        chunks = rag.retrieve("What is the leave policy?", k=k)
        fitted = fit_context(chunks, budget=100)
        words = sum(len(c["text"].split()) for c in fitted)
        print(f"k={k}: selected={len(fitted)}, words={words}, budget=100")

if __name__ == "__main__":
    main()
