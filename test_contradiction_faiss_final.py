import os

os.environ["TOKENIZERS_PARALLELISM"] = "True"

import torch.nn.functional as F

from torch import Tensor

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
)

import random
from datetime import datetime

import torch
import numpy as np

from dataclasses import dataclass, field

from typing import Optional, List, Dict

from sparsecl.models import our_BertForCL
from sparsecl.gte.modeling import NewModelForCL
from multiprocessing import Pool

MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


from beir import LoggingHandler
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import os
import pickle
from tqdm import tqdm
import math
import copy
import faiss
from torch.utils.data import DataLoader

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
    algo_type: Optional[str] = field(
        default="None",
        metadata={
            "help": "which type of model you are using"
        },
    )
    metric: Optional[str] = field(
        default="cos",
        metadata={
            "help": "which metric are you using"
        },
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "how many times of reference data used"
        },
    )
    alpha: float = field(
        default=None,
        metadata={
            "help": "parameter for sparsity metric"
        }
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "which dataset are you using"
        },
    )
    split: Optional[str] = field(
        default="test",
        metadata={
            "help": "which split are you using"
        },
    )



#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

parser = HfArgumentParser(ModelArguments)
model_args = parser.parse_args_into_dataclasses()

dataset_name=model_args[0].dataset_name

if model_args[0].split is None:
    split = "test"
else:
    split=model_args[0].split

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# either model name or model path is specified
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

algo_type=model_args[0].algo_type
max_seq_length=model_args[0].max_seq_length
alpha=model_args[0].alpha
sort_metric=model_args[0].metric

print("max_seq_length=", max_seq_length)

print(model_name)
print(our_model_path)
print(cos_model_name)
print(cos_model_path)

random.seed(211)

folder_path = write_path

# Check if the folder already exists
if not os.path.exists(folder_path):
    # Create the folder
    os.makedirs(folder_path)

output_file=open(os.path.join(write_path,f"{dataset_name}_parallel_output"),"w")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def sentence_embedding(model_name,input_texts,model_path=None):
    print(f"using model name: {model_name}")
    if "our_gte" in model_name:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        print("model path", model_path)

        tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)

        model = NewModelForCL.from_pretrained(
                model_path,
                model_args=model_args[0],
                config=config,
                add_pooling_layer=True,
                trust_remote_code=True,
            )

        model=model.to(device)
        model = torch.nn.DataParallel(model)
        batch_size=64

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

        raw_embeddings = torch.cat(outputs).cpu()
        # print(raw_embeddings.shape)
    elif "our_bge" in model_name or "our_uae" in model_name:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

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
        batch_size=64

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

        raw_embeddings = torch.cat(outputs).cpu()
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
        batch_size=64

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

        raw_embeddings = torch.cat(outputs).cpu()
    elif model_name=="uae":
        from angle_emb import AnglE

        angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
        batch_size=64
        raw_embeddings=torch.zeros(len(input_texts), 1024)
        with torch.no_grad():
            for i in tqdm(range(0,len(input_texts),batch_size)):
                mini_embeddings = angle.encode(input_texts[i:min(i+batch_size,len(input_texts))])
                raw_embeddings[i:min(i+batch_size,len(input_texts))]=torch.from_numpy(mini_embeddings)

    # normalize
    raw_embeddings=raw_embeddings/torch.norm(raw_embeddings, dim=1, keepdim=True)
    raw_embeddings=raw_embeddings.cpu()

    return raw_embeddings

sim_qrels=None
gen_model_name="gpt4"

print(dataset_name, split)
corpus={}
for corpus_split in ["train","dev","test"]:
    if "arguana" in dataset_name:
        read_path=f"./data/arguana_{corpus_split}_retrieval_final.pkl"
    else:
        read_path=f"./data/{dataset_name}_{corpus_split}_retrieval_{gen_model_name}_final.pkl"
    with open(read_path,"rb") as f:
        split_corpus=pickle.load(f)
    corpus={**corpus,**split_corpus}
if "arguana" in dataset_name:
    read_path=f"./data/arguana_{split}_retrieval_final.pkl"
else:
    read_path=f"./data/{dataset_name}_{split}_retrieval_{gen_model_name}_final.pkl"
with open(read_path,"rb") as f:
    _=pickle.load(f)
    queries=pickle.load(f)
    qrels=pickle.load(f)

no_title=True

input_texts=[]
passage_id=[]
passage_name={}

for pid, value in corpus.items():

    passage=value.get("text", "").strip()
    passage_id.append(pid)
    passage_name[pid]=len(input_texts)
    input_texts.append(passage)

query_texts=[]
query_map={}
for qid, value in queries.items():
    passage=value.strip()
    query_map[qid]=len(query_texts)
    query_texts.append(passage)

cos_input_embeddings=sentence_embedding(cos_model_name,input_texts,cos_model_path)
cos_input_embeddings=cos_input_embeddings.to(torch.float32).numpy()

cos_query_embeddings=sentence_embedding(cos_model_name,query_texts,cos_model_path)
cos_query_embeddings=cos_query_embeddings.to(torch.float32).numpy()

if "both" in sort_metric:
    input_embeddings=sentence_embedding(model_name,input_texts,our_model_path)
    input_embeddings=input_embeddings.to(torch.float32).numpy()

    query_embeddings=sentence_embedding(model_name,query_texts,our_model_path)
    query_embeddings=query_embeddings.to(torch.float32).numpy()
    sqrt_dim=math.sqrt(input_embeddings.shape[1])

if not os.path.exists("./indices"):
    os.makedirs("./indices")

# build KNN index for embeddings
if cos_model_path is not None:
    folders=os.path.normpath(cos_model_path).split("/")
    index_name=os.path.join("./indices",dataset_name+"-"+folders[-1]+".faiss")
else:
    index_name=os.path.join("./indices",dataset_name+"-"+cos_model_name+".faiss")

print("index name", index_name)

if os.path.exists(index_name):
    index = faiss.read_index(index_name)
    print("index load")
else:
    dim=cos_input_embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim,64)
    index.add(cos_input_embeddings)
    faiss.write_index(index, index_name)
    print("index completes")

# first search for the top 1000 passages based on cosine similarity, and then rerank them by combining cosine and sparsity
index.hnsw.efSearch = 1000
k_neighbors=1000
D,ids=index.search(cos_query_embeddings,k=k_neighbors)

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
            passage_name=passage_id[pid]
            score.append((dist(cos_input_embeddings[pid],cos_query_embeddings[qid],"cos"),passage_name))
        cos_results[query_name]={}
        for i in range(10):
            cos_results[query_name][score[i][1]]=float(score[i][0])
    elif "both_sum" in sort_metric:
        score=[]
        for i in range(k_neighbors):
            pid=ids[qid][i]
            passage_name=passage_id[pid]
            score.append((dist(cos_input_embeddings[pid],cos_query_embeddings[qid],"cos"),passage_name,dist(input_embeddings[pid],query_embeddings[qid],second_metric)))

        cos_results[query_name]={}
        for i in range(10):
            cos_results[query_name][score[i][1]]=float(score[i][0])

        score=sorted(score,key=lambda x: float(x[0]+alpha*x[2]),reverse=True)
        results[query_name]={}
        for i in range(10):
            results[query_name][score[i][1]]=float(score[i][0]+alpha*score[i][2])
    return (results,cos_results)


def retrieval_test():

    query_args=[]
    for query_name in qrels:
        query_args.append(query_name)

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

    ret=0
    if results!={}:
        print("sparsity contradiction retrieval results")
        print("sparsity contradiction retrieval results",file=output_file)
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, [1,10])
        print(ndcg,recall)
        print(ndcg,recall,file=output_file)
        ret=ndcg["NDCG@10"]
        if sim_qrels is not None:
            print("sparsity similarity retrieval results")
            print("sparsity similarity retrieval results",file=output_file)
            ndcg, _map, recall, precision = retriever.evaluate(sim_qrels, results, [1,10])
            print(ndcg,recall)
            print(ndcg,recall,file=output_file)

    if cos_results!={}:
        print("cos contradiction retrieval results")
        print("cos contradiction retrieval results",file=output_file)
        ndcg, _map, recall, precision = retriever.evaluate(qrels, cos_results, [1,10])
        print(ndcg,recall)
        print(ndcg,recall,file=output_file)
        if sim_qrels is not None:
            print("cos similarity retrieval results")
            print("cos similarity retrieval results",file=output_file)
            ndcg, _map, recall, precision = retriever.evaluate(sim_qrels, cos_results, [1,10])
            print(ndcg,recall)
            print(ndcg,recall,file=output_file)

    print(ret,file=output_file)
    return ret

def hyper_parameter_selection(interval,bins=10,eps=0.01):
    global alpha

    l=interval[0]
    r=interval[1]
    while r-l>eps:
        optimal_value=0
        optimal_interval=(0,0)
        for i in range(bins):
            alpha=l+(i+0.5)*(r-l)/bins
            print(alpha)
            print(alpha,file=output_file)
            value=retrieval_test()
            if value>optimal_value:
                optimal_value=value
                optimal_interval=(l+i*(r-l)/bins,l+(i+1)*(r-l)/bins)
        l=optimal_interval[0]
        r=optimal_interval[1]
    return l


second_metric="hoyer"

print(second_metric)

if __name__ == '__main__':
    if model_args[0].alpha is not None:
        alpha_choice=model_args[0].alpha
    else:
        alpha_choice=None
        if split=="test":
            dev_write_path=write_path.replace("test_"+dataset_name,"dev_"+dataset_name)
            if os.path.exists(os.path.join(dev_write_path,f"{dataset_name}_parallel_output")):
                with open(os.path.join(dev_write_path,f"{dataset_name}_parallel_output"),"r") as file:
                    for line in file:
                        # print(line)
                        if "final alpha" in line:
                            parts = line.split()
                            print("read alpha choice from dev file")
                            alpha_choice=float(parts[2])
                            break
        if alpha_choice is None:
            alpha_choice=hyper_parameter_selection((0,10),bins=10,eps=0.01)
            print("hyper paramter selection")

    alpha=alpha_choice
    print("final alpha",alpha)
    print("final alpha",alpha,file=output_file)

    retrieval_test()
