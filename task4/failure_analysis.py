"""Template for classifying the three worst evaluation cases."""
FAILURES=[
 {'question':'Example question 1','failure_type':'retrieval','evidence':'Expected policy chunk was not retrieved','fix':'Improve chunking or retrieval query'},
 {'question':'Example question 2','failure_type':'generation','evidence':'Context contained answer but response was unsupported','fix':'Strengthen grounded prompt and abstention'},
 {'question':'Example question 3','failure_type':'retrieval','evidence':'Relevant source ranked too low','fix':'Use hybrid retrieval and reranking'},
]
if __name__=='__main__':
    for item in FAILURES: print(item)
    print('Replace these illustrative cases with the three worst cases from your actual evaluation.')
