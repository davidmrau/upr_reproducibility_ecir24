from datasets import load_dataset, Dataset
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, T5ForConditionalGeneration
from collections import defaultdict
from metrics import Trec
import os
import sys

# loading the trec run into a list
def load_trec_run(fname):
    trec_run_data = []
    for line in open(fname):
        parts = line.strip().split()
        qid = parts[0]
        did = parts[2]
        trec_run_data.append((qid, did))
    return trec_run_data


# hf dataset to a dict mapping _id -> text
def ds_to_map(dataset, split_name):
    print(dataset, split_name)
    examples = dataset[split_name]
    mapping = {}
    for example in tqdm(examples):
        if 'title' in example:
            content = example['title'] + " " + example['text']
        else:
            content = example['text']
        mapping[example["_id"]] =  content
    return mapping

# this is a wrapper function taking trec_run, corpus, and queries as input
# returning a HF dataset
def make_hf_dataset(trec_run, corpus, queries):
    data = defaultdict(list)
    for qid, did in trec_run:
        query, doc = queries[qid], corpus[did]
        data['query'].append(query)
        data['document'].append(doc)
        data['did'].append(did)
        data['qid'].append(qid)
    return Dataset.from_dict(data)

# helper to load qrels into correct format
def trectools_qrel(qrels_name):
    qrels_hf = load_dataset(qrels_name)
    data = qrels_hf['test']
    return {'query': data['query-id'], 'docid': data['corpus-id'], 'rel': data['score']}

# write qrels to files that are compatible with trec_eval
def dump_qrels(dataset_name, hf_dataset, folder='qrels'):
    qrels_file = f'{folder}/qrels_{dataset_name}.txt'
    with open(qrels_file, 'w') as fout:
        for el in hf_dataset:
            score = el['score']
            did = el['corpus-id']
            qid = el['query-id']
            fout.write(f'{qid}\t0\t{did}\t{score}\n')

    return qrels_file

# defining the instruction prompt
def format_instruction(sample):
    return f"Passage: {sample['document']}. Please write a question based on this passage."

# loading the model and tokenizer
def load_model_and_tokenizer(model_name):
    model = T5ForConditionalGeneration.from_pretrained(model_name, device_map='auto', torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='right',trust_remote_code=True)
    return model, tokenizer

# custom cullate function
def collate_fn(batch):
    qids = [sample['qid'] for sample in batch]
    dids = [sample['did'] for sample in batch]
    # target is Question : {question}
    target = ['Question: ' + sample['query'] for sample in batch]
    # format the model input using the format template
    instr = [format_instruction(sample)  for sample in batch]  # Add prompt to each text
    # tokenize
    instr_tokenized = tokenizer(instr, padding=True, truncation=True, return_tensors="pt", max_length=512)
    target_tokenized = tokenizer(target, padding=True, max_length=128, truncation=True, return_tensors="pt", add_special_tokens=False).input_ids
    return qids, dids, instr_tokenized, target_tokenized


# applies forward and calculates neg. query-likelihood 
def get_scores(model, instr_tokenized, target_tokenized):
    logits = model(input_ids=instr_tokenized.to('cuda').input_ids, attention_mask=instr_tokenized.to('cuda').attention_mask, labels=target_tokenized.to('cuda')).logits
    log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
    loss = log_softmax.gather(2, target_tokenized.unsqueeze(2)).squeeze(2)
    mask = (target_tokenized != 0).float()
    loss = loss * mask
    loss = torch.sum(loss, dim=1)
    return loss



def rerank(dataset_name, model, tokenizer, bm25_runs, batch_size, ranking_file, hf_user, template_run_file):
    #load qrels
    qrels_hf = load_dataset(f'{hf_user}/{dataset_name}-qrels')
    split = 'validation' if 'dataset_name' == 'msmarco' else 'test'
    # write qrels to folder 
    qrels_file = dump_qrels(dataset_name, qrels_hf[split], folder='qrels')


    # load queries, corpus from hf hub
    # change to username hub (redated for anonymity reasons for submission)
    if 'username' == hf_user:
        queries = ds_to_map(load_dataset(f'{hf_user}/{dataset_name}'), 'queries')
        corpus = ds_to_map(load_dataset(f'{hf_user}/{dataset_name}'), 'corpus')
    else:
        queries = ds_to_map(load_dataset(f'{hf_user}/{dataset_name}', 'queries'), 'queries')
        corpus = ds_to_map(load_dataset(f'{hf_user}/{dataset_name}', 'corpus'), 'corpus')
    # load trec run
    trec_run = load_trec_run(bm25_runs + '/' + template_run_file.format(dataset_name))

    # load qrels
    qrels = trectools_qrel(f'{hf_user}/{dataset_name}-qrels')

    dataset = make_hf_dataset(trec_run, corpus, queries)
    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn,  num_workers=4)

    # do actual infrence
    res_test = defaultdict(dict)
    with torch.inference_mode():
        for batch_inp in tqdm(dataloader):
            qids, dids, instr, target = batch_inp
            # to cuda
            instr_tokenized= instr.to('cuda')
            target_tokenized = target.to('cuda')
            # get scores
            scores = get_scores(model, instr_tokenized, target_tokenized)
            batch_num_examples = scores.shape[0]
            # for each example in batch write into dict
            for i in range(batch_num_examples):
                res_test[qids[i]][dids[i]] = scores[i].item()
    sorted_scores = []
    q_ids = []
    # for each query sort after scores
    for qid, docs in res_test.items():
        sorted_scores_q = [(doc_id, docs[doc_id]) for doc_id in sorted(docs, key=docs.get
    , reverse=True)]
        q_ids.append(qid)
        sorted_scores.append(sorted_scores_q)

    # run trec_eval
    test = Trec('ndcg_cut_10', 'trec_eval', qrels_file, 100, ranking_file_path=ranking_file)
    eval_score = test.score(sorted_scores, q_ids)
    print('ndcg_cut_10', eval_score)



if __name__ == "__main__":

    # dataset name and batch size is passed as argument
    dataset_name = sys.argv[1]
    batch_size = int(sys.argv[2])
    bm25_runs = sys.argv[3] 
    retriever = 'bm25' if 'bm25' in bm25_runs else 'qld'
    template_run_file = f"run.beir.{retriever}-" + "multifield.{}.txt" 
    print(template_run_file)

    
    # define inputs
    model_name = 'bigscience/T0_3B'
    print('loading model')
    model, tokenizer = load_model_and_tokenizer(model_name)


    print(dataset_name)
    # define output
    ranking_file = f'reranking/{bm25_runs}_{dataset_name}_{model_name.replace("/", "_")}'
    if not os.path.exists(ranking_file):
        print(ranking_file)
        # change to username hub (redated for anonymity reasons for submission)
        hf_user = 'username' if 'trec_dl' in dataset_name or 'cqadupstack' in dataset_name  else 'BeIR'
        if hf_user == 'username':
            print(f'{dataset_name} is not accessible on the hub due to preserving anonymity of the submission, this will change upon acceptence.' )
            exit()
        rerank(dataset_name, model, tokenizer, bm25_runs, batch_size, ranking_file, hf_user, template_run_file)
