from datasets import load_dataset, Dataset
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, T5ForConditionalGeneration
from collections import defaultdict
from metrics import Trec
import os
import sys



def load_trec_run(fname):
    trec_run_data = []
    for line in open(fname):
        parts = line.strip().split()
        qid = parts[0]
        did = parts[2]
        trec_run_data.append((qid, did))
    return trec_run_data


# to dict
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

def make_hf_dataset(trec_run, corpus, queries):
    data = defaultdict(list)
    for qid, did in trec_run:
        query, doc = queries[qid], corpus[did]
        data['query'].append(query)
        data['document'].append(doc)
        data['did'].append(did)
        data['qid'].append(qid)
    return Dataset.from_dict(data)


def trectools_qrel(qrels_name):
    qrels_hf = load_dataset(qrels_name)
    data = qrels_hf['test']
    return {'query': data['query-id'], 'docid': data['corpus-id'], 'rel': data['score']}

def dump_qrels(dataset_name, hf_dataset, folder='qrels'):
    qrels_file = f'{folder}/qrels_{dataset_name}.txt'
    with open(qrels_file, 'w') as fout:
        for el in hf_dataset:
            score = el['score']
            did = el['corpus-id']
            qid = el['query-id']
            fout.write(f'{qid}\t0\t{did}\t{score}\n')

    return qrels_file

def load_model_and_tokenizer(model_name):
    try:
        quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype='float16',
        bnb_4bit_use_dobule_quant=False
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', quantization_config=quant_config, use_flash_attention_2=True)
    except:
        try:
            print('- ' * 10 + ' Quantization and Flash Attention  2.0 not used! ' + '- ' * 10)
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
        except:
            print('- ' * 10 + 'Using T5 model' +  '- ' * 10)
            model = T5ForConditionalGeneration.from_pretrained(model_name, device_map='auto', torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='right',trust_remote_code=True)
    return model, tokenizer


def collate_fn(batch, prompt):
    qids = [sample['qid'] for sample in batch]
    dids = [sample['did'] for sample in batch]
    #target = [sample['gen_rel_document'] for sample in batch]
    target = ['Question: ' + sample['query'] for sample in batch]
    instr = [prompt.format(sample['document'])  for sample in batch]  # Add prompt to each text
    instr_tokenized = tokenizer(instr, padding=True, truncation=True, return_tensors="pt", max_length=512)
    target_tokenized = tokenizer(target, padding=True, max_length=128, truncation=True, return_tensors="pt", add_special_tokens=False).input_ids
    return qids, dids, instr_tokenized, target_tokenized




def get_scores(model, instr_tokenized, target_tokenized):
    logits = model(**instr_tokenized.to('cuda')).logits
    #loss_fct = CrossEntropyLoss(reduction='none', ignore_index=model.config.pad_token_id)
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    target = target_tokenized.to('cuda')
    logits_target = logits[:, -(target_tokenized.shape[1]+1):-1 :].permute(0, 2, 1)
    loss = loss_fct(logits_target, target)
    return -loss.mean(1).unsqueeze(1)



def get_scores(model, instr_tokenized, target_tokenized):
    logits = model(input_ids=instr_tokenized.to('cuda').input_ids, attention_mask=instr_tokenized.to('cuda').attention_mask, labels=target_tokenized.to('cuda')).logits
    log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
    loss = log_softmax.gather(2, target_tokenized.unsqueeze(2)).squeeze(2)
    mask = (target_tokenized != 0).float()
    loss = loss * mask
    loss = torch.sum(loss, dim=1)
    return loss



def rerank(dataset_name, model, tokenizer, bm25_runs, batch_size, ranking_file, hf_user, prompt):
    #load data
    template_run_file = "run.beir.bm25-multifield.{}.txt" 
    qrels_hf = load_dataset(f'{hf_user}/{dataset_name}-qrels')
    split = 'validation' if 'dataset_name' == 'msmarco' else 'test'
    qrels_file = dump_qrels(dataset_name, qrels_hf[split], folder='qrels')
    if 'username' == hf_user:
        queries = ds_to_map(load_dataset(f'{hf_user}/{dataset_name}'), 'queries')
        corpus = ds_to_map(load_dataset(f'{hf_user}/{dataset_name}'), 'corpus')
    else:
        queries = ds_to_map(load_dataset(f'{hf_user}/{dataset_name}', 'queries'), 'queries')
        corpus = ds_to_map(load_dataset(f'{hf_user}/{dataset_name}', 'corpus'), 'corpus')
    trec_run = load_trec_run(bm25_runs + '/' + template_run_file.format(dataset_name))

    qrels = trectools_qrel(f'{hf_user}/{dataset_name}-qrels')

    dataset = make_hf_dataset(trec_run, corpus, queries)
    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: collate_fn(batch, prompt),  num_workers=4)

    res_test = defaultdict(dict)
    with torch.inference_mode():
        for batch_inp in tqdm(dataloader):
            qids, dids, instr, target = batch_inp
            instr_tokenized= instr.to('cuda')
            target_tokenized = target.to('cuda')
            scores = get_scores(model, instr_tokenized, target_tokenized)
            batch_num_examples = scores.shape[0]
            # for each example in batch
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

    test = Trec('ndcg_cut_10', 'trec_eval', qrels_file, 100, ranking_file_path=ranking_file)
    eval_score = test.score(sorted_scores, q_ids)
    print('ndcg_cut_10', eval_score)



dataset_name = sys.argv[1]

prompts = [

"Passage: {}. Please write a question based on this passage.", #default
"Please write a question based on the passage. Passage: {}.", # passage at the end
"Passage: {}. Please write a question based on the previous input.", # not tagging passage
"Passage: {}. Write a question based on this passage.", # leave out please
"passage: {}. please write a question based on this passage.", # lower case
"Text: {}. Please write a question based on this text.", # text instead of passage
"Passage: {}. Please write a question that this passage could answer.", # different formulation
"{}", # not instr
"We are using zero-shot question generation to re-rank passages based on the likelihood of the generated question. For this please write a question based on this passage. Passage: {}.", # info about task
"Passage: {}.", #default
"Passage: {}. Write question.", # different formulation
"{}. Please write a question based on this passage.", # not tagging passage
"Passage: {}. Please generate a question based on this passage.", #default
"Passage: {}. Please output a question based on this passage.", #default
"Passage: {}. Please write a query based on this passage.", #default
]

print(len(prompts))
p2id = {i:p for  i, p in enumerate(prompts)}
model_name = 'bigscience/T0_3B'
model, tokenizer = load_model_and_tokenizer(model_name)
batch_size = 192
bm25_runs = "beir_bm25_runs_top100"


for _id in p2id:
    if _id <=11 or _id == 14:
        continue
    print(_id)
    ranking_file = f'reranking_prompts/{bm25_runs}_{dataset_name}_{model_name.replace("/", "_")}_prompt_{_id}'
    if not os.path.exists(ranking_file):
        hf_user = 'username' if 'trec_dl' in dataset_name or 'cqadupstack' in dataset_name  else 'BeIR'
        rerank(dataset_name, model, tokenizer, bm25_runs, batch_size, ranking_file, hf_user, p2id[_id])
