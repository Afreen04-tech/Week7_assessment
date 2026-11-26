# Business Summary – RAG Policy Assistant

The prototype organizes company policies into searchable chunks and retrieves relevant passages before drafting an answer. This helps employees find policy information faster while reducing unsupported answers through a retrieval threshold and a clear abstention response. Answers should include source documents and page references so employees can verify the policy.

Evaluation should compare faithfulness, answer relevancy, context precision and context recall on the 15-question golden dataset. Record actual baseline scores, inspect the three weakest questions, classify each as a retrieval or generation failure, apply a targeted fix, and rerun the same dataset. Do not report example scores as measured results. The business target is faithfulness above 0.85; the team should only claim that target after the rerun confirms it.
