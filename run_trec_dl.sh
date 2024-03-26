DATASETS=('trec_dl20' 'trec_dl19')
BATCH_SIZE=96

for DATASET in $DATASETS; do
    python3 rerank_t5.py $DATASET $BATCH_SIZE 
done
