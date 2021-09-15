1. 2021-09-04-10-38-07	first_incompletedataset_norewrite.json

- No rewrite (using last 3 utterances: question + model answer + current question)
- BM25 retrieval
- Pegasus for generation (trained on 10% of QReCC dataset)

2. 2021-09-04-10-39-42	usingtruthrewrite_incompletedataset.json

- Using truth rewrite (using last 3 utterances: question + model answer + current question)
- BM25 retrieval
- Pegasus for generation (trained on 10% of QReCC dataset)

3. 2021-09-06-09-21-43	usingtruthrewrite.json

- Using truth rewrite (using last 3 utterances: question + model answer + current question)
- BM25 retrieval
- Pegasus for generation (trained on the entire QReCC dataset)

4. 2021-09-08-07-07-57	test_rewriting_sparse.json

- T5 for rewriting (using last 5 utterances: model rewritte and model answers)
- BM25 retrieval
- Pegasus for generation (trained on the entire QReCC dataset)

5. 2021-09-08-07-09-57	test_rewriting_nohistory_sparse.json

- T5 for rewriting (using last 5 utterances: model rewritte and model answers)
- BM25 retrieval (using only current model rewritten question)
- Pegasus for generation (trained on the entire QReCC dataset)

6. 2021-09-08-07-09-57	better_test_rewriting_nohistory_sparse.json

- T5 for rewriting (using last 5 utterances: model rewritten and model answers)
- BM25 retrieval (using only current model rewritten question)
- Pegasus for generation (trained on the entire QReCC dataset with truth rewrite)

7. 2021-09-08-21-49-44	better_test_rewriting_nohistory_sparse_norepetition.json

- T5 for rewriting (using last 5 utterances: model rewritten and model answers)
- BM25 retrieval (using only current model rewritten question)
- Pegasus for generation (no_repeat_ngram_size=10, trained on the entire QReCC dataset with truth rewrite)

8. question.json

- T5 for rewriting (using all the last utterances: questions)
- BM25 retrieval (using only current model rewritten question)
- Pegasus for generation (no_repeat_ngram_size=10, trained much longer)

9. answer.json

- T5 for rewriting (using all the last utterances: questions and model answers)
- BM25 retrieval (using only current model rewritten question)
- Pegasus for generation (no_repeat_ngram_size=10, trained much longer)

10. rewrite.json

- T5 for rewriting (using all the last utterances: model rewritten)
- BM25 retrieval (using only current model rewritten question)
- Pegasus for generation (no_repeat_ngram_size=10, trained much longer)

11. rewrite_and_answer.json

- T5 for rewriting (using all the last utterances: model rewritten and model answers)
- BM25 retrieval (using only current model rewritten question)
- Pegasus for generation (no_repeat_ngram_size=10, trained much longer)
