import torch.nn.functional as F

from torch import Tensor

from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
)

import torch
import numpy as np
import os

from dataclasses import dataclass, field

from typing import Optional, List, Dict
from sparsecl.models import our_BertForCL
from multiprocessing import Pool
import random
from datetime import datetime

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)




from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import pickle
from tqdm import tqdm

import math
import faiss
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "True"

# Get current time
current_time = datetime.now().time()
formatted_time = current_time.strftime('%H:%M:%S')


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    temp: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for softmax."
        }
    )
    pooler_type: str = field(
        default="cls",
        metadata={
            "help": "What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last)."
        }
    )
    hard_negative_weight: float = field(
        default=0,
        metadata={
            "help": "The **logit** of weight for hard negatives (only effective if hard negatives are used)."
        }
    )
    do_mlm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use MLM auxiliary objective."
        }
    )
    mlm_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight for MLM auxiliary objective (only effective if --do_mlm)."
        }
    )
    mlp_only_train: bool = field(
        default=False,
        metadata={
            "help": "Use MLP only during training"
        }
    )
    write_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "where to write test result"
        },
    )
    eval_task: Optional[str] = field(
        default=None,
        metadata={
            "help": "which task to eval"
        },
    )
    cos_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    cos_model_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "which type of model you are using"
        },
    )
    model_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "which type of model you are using"
        },
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "how many times of reference data used"
        },
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "which dataset are you using"
        },
    )




#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download scifact.zip dataset and unzip the dataset
# url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
# out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
# data_path = util.download_and_unzip(url, out_dir)

parser = HfArgumentParser(ModelArguments)
model_args = parser.parse_args_into_dataclasses()

dataset_name=model_args[0].dataset_name

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

model_name=model_args[0].model_name
cos_model_name=model_args[0].cos_model_name

our_model_path=model_args[0].model_name_or_path
cos_model_path=model_args[0].cos_model_name_or_path

if our_model_path is not None:
    if "bge" in our_model_path.lower():
        model_name="our_bge"
    elif "uae" in our_model_path.lower():
        model_name="our_uae"
    elif "gte" in our_model_path.lower():
        model_name="our_gte"
    config = AutoConfig.from_pretrained(our_model_path,trust_remote_code=True,)
if cos_model_path is not None:
    if "bge" in cos_model_path.lower():
        cos_model_name="our_bge"
    elif "uae" in cos_model_path.lower():
        cos_model_name="our_uae"
    elif "gte" in cos_model_path.lower():
        cos_model_name="our_gte"
    config = AutoConfig.from_pretrained(cos_model_path,trust_remote_code=True,)
write_path=model_args[0].write_path

max_seq_length=model_args[0].max_seq_length

random.seed(211)

num_para=3
max_para=3

folder_path = write_path

# Check if the folder already exists
if not os.path.exists(folder_path):
    # Create the folder
    os.makedirs(folder_path)

output_file=open(os.path.join(write_path,f"test_data_cleaning_{dataset_name}_parallel_output"),"w")

print("max_seq_length=", max_seq_length)
print(model_name)
print(our_model_path)
print(cos_model_name)
print(cos_model_path)

print("max_seq_length=", max_seq_length,file=output_file)
print(model_name,file=output_file)
print(our_model_path,file=output_file)
print(cos_model_name,file=output_file)
print(cos_model_path,file=output_file)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def sentence_embedding(model_name,input_texts,model_path=None):
    print(f"using model name: {model_name}")
    if "our_gte" in model_name:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        # device = "cpu"

        print("model path", model_path)

        tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
        from sparsecl.gte.modeling import NewModelForCL
        model = NewModelForCL.from_pretrained(
                model_path,
                model_args=model_args[0],
                config=config,
                add_pooling_layer=True,
                trust_remote_code=True,
            )

        model=model.to(device)
        model = torch.nn.DataParallel(model)
        batch_size=128
        # inputs = [tokenizer(text, max_length=max_seq_length, padding=True, truncation=True, return_tensors='pt') for text in input_texts]

        # Create DataLoader for batching
        data_loader = DataLoader(input_texts, batch_size=batch_size)

        # Perform inference batch by batch
        outputs = []
        model.eval()
        with torch.no_grad():
            for batch_texts in tqdm(data_loader):
                batch_inputs = tokenizer(batch_texts, max_length=max_seq_length, padding=True, truncation=True, return_tensors='pt')
                batch_inputs = {key: value.to(device) for key, value in batch_inputs.items()}
                batch_outputs = model(**batch_inputs,output_hidden_states=True, return_dict=True, sent_emb=True)
                outputs.append(batch_outputs.pooler_output)

        outputs_cpu = [output.cpu() for output in outputs]
        raw_embeddings = torch.cat(outputs_cpu)
        # raw_embeddings = torch.cat(outputs).cpu()
        print(raw_embeddings.shape)
    elif "our_bge" in model_name or "our_uae" in model_name:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        # device = "cpu"

        print(torch.cuda.device_count())
        print("model path", model_path)

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        model = our_BertForCL.from_pretrained(
                    model_path,
                    from_tf=False,
                    config=config,
                    use_auth_token=None,
                    model_args=model_args[0]
                )

        model=model.to(device)
        model = torch.nn.DataParallel(model)
        batch_size=128
        # inputs = [tokenizer(text, max_length=max_seq_length, padding=True, truncation=True, return_tensors='pt') for text in input_texts]

        # Create DataLoader for batching
        data_loader = DataLoader(input_texts, batch_size=batch_size)

        # Perform inference batch by batch
        outputs = []
        model.eval()
        with torch.no_grad():
            for batch_texts in tqdm(data_loader):
                batch_inputs = tokenizer(batch_texts, max_length=max_seq_length, padding=True, truncation=True, return_tensors='pt')
                batch_inputs = {key: value.to(device) for key, value in batch_inputs.items()}
                batch_outputs = model(**batch_inputs,output_hidden_states=True, return_dict=True, sent_emb=True)
                outputs.append(batch_outputs.pooler_output)

        outputs_cpu = [output.cpu() for output in outputs]
        raw_embeddings = torch.cat(outputs_cpu)
        # raw_embeddings = torch.cat(outputs).cpu()
        # print(raw_embeddings.shape)
    elif "gte" in model_name or "bge" in model_name:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        if "bge" in model_name:
            dim=768
            tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5')
            model = AutoModel.from_pretrained('BAAI/bge-base-en-v1.5')
        elif "gte" in model_name:
            dim=1024
            tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
            model = AutoModel.from_pretrained('Alibaba-NLP/gte-large-en-v1.5',trust_remote_code=True)

        model=model.to(device)
        model = torch.nn.DataParallel(model)
        model.eval()
        batch_size=128

        data_loader = DataLoader(input_texts, batch_size=batch_size)
        # Perform inference batch by batch
        outputs = []
        with torch.no_grad():
            for batch_texts in tqdm(data_loader):
                batch_inputs = tokenizer(batch_texts, max_length=max_seq_length, padding=True, truncation=True, return_tensors='pt')
                batch_inputs = {key: value.to(device) for key, value in batch_inputs.items()}
                batch_outputs = model(**batch_inputs,output_hidden_states=True, return_dict=True)

                if "gte" in model_name:
                    mini_embeddings = batch_outputs.last_hidden_state[:, 0]
                elif "bge" in model_name:
                    mini_embeddings=batch_outputs[0][:, 0]

                mini_embeddings = torch.nn.functional.normalize(mini_embeddings, p=2, dim=1)
                outputs.append(mini_embeddings)

        outputs_cpu = [output.cpu() for output in outputs]
        raw_embeddings = torch.cat(outputs_cpu)
        # raw_embeddings = torch.cat(outputs).cpu()
    elif model_name=="uae":
        from angle_emb import AnglE

        angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
        batch_size=128
        raw_embeddings=torch.zeros(len(input_texts), 1024)
        with torch.no_grad():
            for i in tqdm(range(0,len(input_texts),batch_size)):
                mini_embeddings = angle.encode(input_texts[i:min(i+batch_size,len(input_texts))])
                raw_embeddings[i:min(i+batch_size,len(input_texts))]=torch.from_numpy(mini_embeddings)

    # normalize
    raw_embeddings=raw_embeddings/torch.norm(raw_embeddings, dim=1, keepdim=True)
    raw_embeddings=raw_embeddings.cpu()

    return raw_embeddings

# cos_input_embeddings=None
# cos_query_embeddings=None
# input_embeddings=None
# query_embeddings=None
# corpus=None
# queries=None
# qrels=None
# sim_qrels=None

wrong_qrels=None

gen_model_name="gpt4"
# gen_model_name="gpt3.5"

if dataset_name in ["msmarco","hotpotqa"]:
    data_file_path=os.path.join("./data",f"{dataset_name}_cleaning_{gen_model_name}_final.pkl")
    with open(data_file_path, 'rb') as file:
        corpus=pickle.load(file)
        queries=pickle.load(file)
        qrels=pickle.load(file)
        wrong_qrels=pickle.load(file)
else:
    print("not implemented")
    exit()

# for qid in list(qrels.keys())[:10]:
#     print(qid, qrels[qid])
#     for pid in qrels[qid]:
#         assert(pid in corpus)

# for query_name in list(queries.keys())[:10]:
#     print(query_name,queries[query_name])

no_title=True

input_texts=[]
input_texts_notitle=[]
passage_id=[]
passage_name={}

passage_corrupted=[False]*len(corpus)
passage_groundtruth=[False]*len(corpus)

for pid, value in corpus.items():

    if no_title is True:
        passage=value.get("text", "").strip()
    else:
        passage=(value.get("title", "") + " " + value.get("text", "")).strip()
    passage_id.append(pid)
    passage_name[pid]=len(input_texts)
    input_texts.append(passage)

query_texts=[]
query_map={}
for qid, value in queries.items():
    passage=value.strip()
    query_map[qid]=len(query_texts)
    query_texts.append(passage)

# augment those "para" as queries
for pid, value in corpus.items():
    if "para" in pid:
        passage=value.get("text", "").strip()
        query_map[pid]=len(query_texts)
        query_texts.append(passage)

print(len(input_texts))
print(len(query_texts))

if not os.path.exists("./embedding_data"):
    os.makedirs("./embedding_data")

cos_embedding_file_name=os.path.join("./embedding_data",f"{dataset_name}_{cos_model_name}.pkl")

if os.path.exists(cos_embedding_file_name):
    with open(cos_embedding_file_name, 'rb') as file:
        cos_input_embeddings=pickle.load(file)
        cos_query_embeddings=pickle.load(file)
else:
    cos_input_embeddings=sentence_embedding(cos_model_name,input_texts,cos_model_path)
    cos_input_embeddings=cos_input_embeddings.to(torch.float32).numpy()
    cos_query_embeddings=sentence_embedding(cos_model_name,query_texts,cos_model_path)
    cos_query_embeddings=cos_query_embeddings.to(torch.float32).numpy()
    with open(cos_embedding_file_name, 'wb') as file:
        pickle.dump(cos_input_embeddings,file)
        pickle.dump(cos_query_embeddings,file)


folders=os.path.normpath(our_model_path).split("/")

our_embedding_file_name=os.path.join("./embedding_data",f"{dataset_name}_{folders[-1]}.pkl")

if os.path.exists(our_embedding_file_name):
    with open(our_embedding_file_name, 'rb') as file:
        input_embeddings=pickle.load(file)
        query_embeddings=pickle.load(file)
else:
    input_embeddings=sentence_embedding(model_name,input_texts,our_model_path)
    input_embeddings=input_embeddings.to(torch.float32).numpy()
    query_embeddings=sentence_embedding(model_name,query_texts,our_model_path)
    query_embeddings=query_embeddings.to(torch.float32).numpy()
    with open(our_embedding_file_name, 'wb') as file:
        pickle.dump(input_embeddings,file)
        pickle.dump(query_embeddings,file)


print(input_embeddings.shape)
print(query_embeddings.shape)

dim=input_embeddings.shape[1]
sqrt_dim=math.sqrt(dim)

print("dim=",dim,"sqrt dim",sqrt_dim)

if not os.path.exists("./indices"):
    os.makedirs("./indices")

if cos_model_path is not None:
    index_name=os.path.join("./indices",f"cleaning_{dataset_name}_{folders[-1]}.faiss")
else:
    index_name=os.path.join("./indices",f"cleaning_{dataset_name}_{cos_model_name}.faiss")

print("index name", index_name)

if os.path.exists(index_name):
    index = faiss.read_index(index_name)
    print("index load")
else:
    index = faiss.IndexHNSWFlat(dim,64)
    index.add(cos_input_embeddings)
    faiss.write_index(index, index_name)
    print("index completes")

index.hnsw.efSearch = 5000
k_neighbors=1000

if cos_model_path is not None:
    search_result_file=os.path.join("./indices",f"cleaning_{dataset_name}_{folders[-1]}_search_L{index.hnsw.efSearch}K{k_neighbors}.pkl")
else:
    search_result_file=os.path.join("./indices",f"cleaning_{dataset_name}_{cos_model_name}_search_L{index.hnsw.efSearch}K{k_neighbors}.pkl")

if os.path.exists(search_result_file):
    with open(search_result_file, 'rb') as file:
        D=pickle.load(file)
        ids=pickle.load(file)
else:
    D,ids=index.search(cos_query_embeddings,k=k_neighbors)
    with open(search_result_file, 'wb') as file:
        pickle.dump(D,file)
        pickle.dump(ids,file)

print(D[:10,:10])
print(ids[:10,:10])

print("search completes!")

model = DRES(models.SentenceBERT("BAAI/bge-base-en-v1.5"), batch_size=16)
retriever = EvaluateRetrieval(model, score_function="cos_sim") # or "cos_sim" for cosine similarity


def dist(A,B,metric="hoyer"):
    if metric=="hoyer":
        diff=A-B
        diff_l1=np.linalg.norm(diff,ord=1)
        diff_l2=np.linalg.norm(diff,ord=2)
        if diff_l2<1e-3:
            return -1e9
        hoyer=(sqrt_dim-diff_l1/diff_l2)/(sqrt_dim-1)
        return hoyer
    elif metric=="cos":
        dot_product = np.dot(A, B)
        norm_a = np.linalg.norm(A)
        norm_b = np.linalg.norm(B)
        cosine_similarity = dot_product / (norm_a * norm_b)
        return cosine_similarity

def process_query(query_name):

    results={}
    cos_results={}
    qid=query_map[query_name]
    # print(qid,query_texts[qid])
    score=[]
    if sort_metric=="cos":
        score=[]
        for i in range(k_neighbors):
            pid=ids[qid][i]
            pid_name=passage_id[pid]
            if test_setting==0:
                if "contra" in pid_name:
                    continue
            elif test_setting==1:
                pass
            elif test_setting==2:
                assert(False)
            elif test_setting==3:
                if passage_corrupted[pid] is True:
                    continue
            score.append((dist(cos_input_embeddings[pid],cos_query_embeddings[qid],"cos"),pid_name,dist(input_embeddings[pid],query_embeddings[qid],"hoyer")))
        score=sorted(score,key=lambda x: x[0],reverse=True)
        cos_results[query_name]={}
        if qid<10:
            print(qid,query_name,query_texts[qid])
        for i in range(10):
            cos_results[query_name][score[i][1]]=max(0,float(score[i][0]))
            if qid<10:
                print(score[i])
    elif sort_metric=="both_sum":
        score=[]
        for i in range(k_neighbors):
            pid=ids[qid][i]
            pid_name=passage_id[pid]
            if test_setting==2:
                if passage_groundtruth[pid] is True or pid_name==query_name:
                    continue
            else:
                assert(False)
            score.append((dist(cos_input_embeddings[pid],cos_query_embeddings[qid],"cos"),pid_name,dist(input_embeddings[pid],query_embeddings[qid],"hoyer")))

        cos_results[query_name]={}
        for i in range(10):
            cos_results[query_name][score[i][1]]=float(score[i][0])


        score=sorted(score,key=lambda x: float(x[0]+alpha*x[2]),reverse=True)
        # score=sorted(score,key=lambda x: float(x[2]),reverse=True)
        results[query_name]={}
        for i in range(10):
            results[query_name][score[i][1]]=float(score[i][0]+alpha*score[i][2])

    return (results,cos_results)

def retrieval_test():

    query_args=[]
    if test_setting!=2:
        for query_name in qrels:
            query_args.append(query_name)
        # query_args=query_args[:100]
    else:
        contra_qrels={}
        original_pids=[]
        for pid in corpus:
            if "para" in pid:
                original_pid=pid[:pid.find("-para")]
                if original_pid not in original_pids:
                    original_pids.append(original_pid)

                    # qid=original_pid+f"-para{random.randint(0,2)}-{gen_model_name}"
                    qid=original_pid+f"-para0-{gen_model_name}"
                    query_args.append(qid)
                    passage_groundtruth[passage_name[qid]]=True
                    contra_qrels[qid]={}
                    for i in range(num_para):
                        contra_pid=original_pid+f"-contra{i}-{gen_model_name}"
                        if contra_pid in corpus:
                            contra_qrels[qid][contra_pid]=1

    with Pool() as pool:
        pool_results = pool.map(process_query, query_args)

    results={}
    cos_results={}
    for res,cos_res in pool_results:
        for key,value in res.items():
            results[key]=value
        for key,value in cos_res.items():
            cos_results[key]=value

    #### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000]

    if test_setting!=2:
        print("cos retrieval results(qrels)")
        print("cos retrieval results(qrels)",file=output_file)
        ndcg, _map, recall, precision = retriever.evaluate(qrels, cos_results, [10])
        print(ndcg,recall)
        print(ndcg,recall,file=output_file)

        if wrong_qrels is not None:
            print("cos retrieval results(contra qrels)")
            print("cos retrieval results(contra qrels)",file=output_file)
            ndcg, _map, recall, precision = retriever.evaluate(wrong_qrels, cos_results, [10])
            print(ndcg,recall)
            print(ndcg,recall,file=output_file)
    else:
        if contra_qrels is not None:
            print("sparsity contradiction retrieval results")
            print("sparsity contradiction retrieval results",file=output_file)
            ndcg, _map, recall, precision = retriever.evaluate(contra_qrels, results, [1,3,10])
            print(ndcg,recall)
            print(ndcg,recall,file=output_file)

        counter=0
        for qid in results:
            keys=list(results[qid].keys())
            for i in range(min(num_para,len(keys))):
                passage_corrupted[passage_name[keys[i]]]=True

# test setting: 0 original 1 corrupted 2 contradiction retrieval 3 after cleaning

print("original testing")
print("original testing",file=output_file)
test_setting=0
sort_metric="cos"
retrieval_test()

print("pre testing")
print("pre testing",file=output_file)
test_setting=1
sort_metric="cos"
retrieval_test()

# Insert alpha paramters from Table3's first line "GTE". We do zeroshot GTE + sparsity enhanced GTE here.
if dataset_name=="msmarco":
    alpha=2.65
elif dataset_name=="hotpotqa":
    alpha=2.36

print("data cleaning")
print("data cleaning",file=output_file)
test_setting=2
sort_metric="both_sum"
retrieval_test()


print("after testing")
print("after testing",file=output_file)
test_setting=3
sort_metric="cos"
retrieval_test()
