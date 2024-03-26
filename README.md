# upr_reproducibility_ecir24
Query Generation using Large Language Models: A reproducibility study of unsupervised passage reranking# Installing packages


To reproduce the UPR results on BeIR first install the requirements.txt using 
```bash
	pip3 install -r requirements.py
```

# Reproducing UPR results on BeIR

Next, make a GPU available in your env. and run the inference script:
```bash
	bash run_beir.sh
```

# Reproducing UPR results on BeIR bm25 top-100

for this experiment run:
```bash
	bash run_beir_top_100.sh
```

# Reproducing UPR results on BeIR QLD top-100

for this experiment run:
```bash
	bash run_beir_qld_top_100.sh
```
# Reproducing UPR results on TREC DL BM25 top-100

for this experiment run:
```bash
	bash run_trec_dl.sh
```
# Reproducing UPR prompting robustness

for this experiment run:
```bash	
	bash run_prompt.sh
```


Cite: 
``
@inproceedings{rau-query-generation,
author = {Rau, David and Kamps, Jaap},
year = {2024},
month = {03},
pages = {226-239},
title = {Query Generation Using Large Language Models: A Reproducibility Study of Unsupervised Passage Reranking},
isbn = {978-3-031-56065-1},
doi = {10.1007/978-3-031-56066-8_19},
booktitle={Advances in Information Retrieval
46th European Conference on Information Retrieval},
}
```
