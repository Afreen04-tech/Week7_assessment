# Task 4 – Golden Dataset and RAGAS

`golden.jsonl` contains 15 questions (8 simple, 4 multi-chunk, 3 unanswerable). Run `python evaluate_ragas.py` to validate and inspect the dataset. For actual RAGAS metric values, run all questions through your Task 1/2 pipeline and supply generated answers, retrieved contexts and references to the RAGAS evaluation API matching your installed version. Update `failure_analysis.py` and `summary.md` with measured results; do not treat example failure rows as measured outcomes.
