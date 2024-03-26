#python -m pyserini.search.lucene \
#  --index beir-v1.0.0-webis-touche2020.multifield \
#  --topics beir-v1.0.0-webis-touche2020-test \
#  --output run.beir.qld-multifield.webis-touche2020.txt \
#  --output-format trec \
#  --batch 36 --threads 12 \
#  --hits 100 --qld --remove-query --fields contents=1.0 title=1.0
#
#
#python -m pyserini.search.lucene \
#  --index beir-v1.0.0-trec-covid.multifield \
#  --topics beir-v1.0.0-trec-covid-test \
#  --output run.beir.qld-multifield.trec-covid.txt \
#  --output-format trec \
#  --batch 36 --threads 12 \
#  --hits 100 --qld --remove-query --fields contents=1.0 title=1.0
#
#
#python -m pyserini.search.lucene \
#  --index beir-v1.0.0-nfcorpus.multifield \
#  --topics beir-v1.0.0-nfcorpus-test \
#  --output run.beir.qld-multifield.nfcorpus.txt \
#  --output-format trec \
#  --batch 36 --threads 12 \
#  --hits 100 --qld --remove-query --fields contents=1.0 title=1.0
#
#python -m pyserini.search.lucene \
#  --index beir-v1.0.0-nq.multifield \
#  --topics beir-v1.0.0-nq-test \
#  --output run.beir.qld-multifield.nq.txt \
#  --output-format trec \
#  --batch 36 --threads 12 \
#  --hits 100 --qld --remove-query --fields contents=1.0 title=1.0
#
#
#python -m pyserini.search.lucene \
#  --index beir-v1.0.0-hotpotqa.multifield \
#  --topics beir-v1.0.0-hotpotqa-test \
#  --output run.beir.qld-multifield.hotpotqa.txt \
#  --output-format trec \
#  --batch 36 --threads 12 \
#  --hits 100 --qld --remove-query --fields contents=1.0 title=1.0
#
#
#python -m pyserini.search.lucene \
#  --index beir-v1.0.0-fiqa.multifield \
#  --topics beir-v1.0.0-fiqa-test \
#  --output run.beir.qld-multifield.fiqa.txt \
#  --output-format trec \
#  --batch 36 --threads 12 \
#  --hits 100 --qld --remove-query --fields contents=1.0 title=1.0
##
#
#python -m pyserini.search.lucene \
#  --index beir-v1.0.0-arguana.multifield \
#  --topics beir-v1.0.0-arguana-test \
#  --output run.beir.qld-multifield.arguana.txt \
#  --output-format trec \
#  --batch 36 --threads 12 \
#  --hits 100 --qld --remove-query --fields contents=1.0 title=1.0
#
#python -m pyserini.search.lucene \
#  --index beir-v1.0.0-quora.multifield \
#  --topics beir-v1.0.0-quora-test \
#  --output run.beir.qld-multifield.quora.txt \
#  --output-format trec \
#  --batch 36 --threads 12 \
#  --hits 100 --qld --remove-query --fields contents=1.0 title=1.0
#
#
#python -m pyserini.search.lucene \
#  --index beir-v1.0.0-dbpedia-entity.multifield \
#  --topics beir-v1.0.0-dbpedia-entity-test \
#  --output run.beir.qld-multifield.dbpedia-entity.txt \
#  --output-format trec \
#  --batch 36 --threads 12 \
#  --hits 100 --qld --remove-query --fields contents=1.0 title=1.0
#
#python -m pyserini.search.lucene \
#  --index beir-v1.0.0-scidocs.multifield \
#  --topics beir-v1.0.0-scidocs-test \
#  --output run.beir.qld-multifield.scidocs.txt \
#  --output-format trec \
#  --batch 36 --threads 12 \
#  --hits 100 --qld --remove-query --fields contents=1.0 title=1.0
#
#
python -m pyserini.search.lucene \
  --index beir-v1.0.0-fever.multifield \
  --topics beir-v1.0.0-fever-test \
  --output run.beir.qld-multifield.fever.txt \
  --output-format trec \
  --batch 36 --threads 12 \
  --hits 100 --qld --remove-query --fields contents=1.0 title=1.0



#python -m pyserini.search.lucene \
#  --index beir-v1.0.0-climate-fever.multifield \
#  --topics beir-v1.0.0-climate-fever-test \
#  --output run.beir.qld-multifield.climate-fever.txt \
#  --output-format trec \
#  --batch 36 --threads 12 \
#  --hits 100 --qld --remove-query --fields contents=1.0 title=1.0

python -m pyserini.search.lucene \
  --index beir-v1.0.0-scifact.multifield \
  --topics beir-v1.0.0-scifact-test \
  --output run.beir.qld-multifield.scifact.txt \
  --output-format trec \
  --batch 36 --threads 12 \
  --hits 100 --qld --remove-query --fields contents=1.0 title=1.0


python -m pyserini.search.lucene \
  --threads 16 --batch-size 128 \
  --index msmarco-v1-passage \
  --topics msmarco-passage-dev-subset \
  --output run.beir.bm25-multifield.msmarco.txt \
  --qld

