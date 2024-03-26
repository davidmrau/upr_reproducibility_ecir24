DATASETS=('scifact' 'scidocs' 'nfcorpus' 'fiqa' 'trec-covid' 'webis-touche2020' 'nq' 'msmarco' 'hotpotqa' 'arguana' 'quora' 'dbpedia-entity' 'fever' 'climate-fever' 'trec_dl20' 'trec_dl19')

DATASETS=('webis-touche2020')
BATCH_SIZE=96
RUNS="beir_qld_runs_top_100"
for DATASET in $DATASETS; do
    python3 rerank_t5.py $DATASET $BATCH_SIZE $RUNS
done
