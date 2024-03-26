# Installing packages

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