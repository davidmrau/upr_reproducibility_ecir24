for sub in android english gaming gis mathematica physics programmers stats tex unix webmasters wordpress;do
	python -m pyserini.search.lucene \
	  --index beir-v1.0.0-cqadupstack-$sub.multifield \
	  --topics beir-v1.0.0-cqadupstack-$sub-test \
	  --output run.beir.qld-multifield.cqadupstack-$sub.txt \
	  --output-format trec \
	  --batch 36 --threads 12 \
	  --hits 100 --qld --remove-query --fields contents=1.0 title=1.0
done
exit
for measure in recall.100 ndcg_cut.10; do
	for sub in android english gaming gis mathematica physics programmers stats tex unix webmasters wordpress;do
		python -m pyserini.eval.trec_eval \
		  -c -m $measure beir-v1.0.0-cqadupstack-$sub-test \
		  run.beir.bm25-multifield.cqadupstack-$sub.txt
	done
done
exit



