DATASETS=('trec_dl20')
BATCH_SIZE=96
RUNS="beir_bm25_runs_top100"
for DATASET in $DATASETS; do
    python3 rerank_t5_prompt.py $DATASET $BATCH_SIZE $RUNS 
done
