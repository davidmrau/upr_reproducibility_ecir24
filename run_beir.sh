DATASETS=('scifact' 'scidocs' 'nfcorpus' 'fiqa' 'trec-covid' 'webis-touche2020' 'nq' 'msmarco' 'hotpotqa' 'arguana' 'quora' 'dbpedia-entity' 'fever' 'climate-fever')

DATASETS=('webis-touche2020')
BATCH_SIZE=96
RUNS='beir_bm25_runs_top1000'
for DATASET in $DATASETS; do
    python3 rerank_t5.py $DATASET $BATCH_SIZE $RUNS
done
