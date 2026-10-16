from saki.paths import release_path
import json
import os
import time
import warnings
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
LOCAL_MODEL_DIR = release_path('${MODEL_ROOT}')
CLAP_WEIGHTS_PATH = os.path.join(LOCAL_MODEL_DIR, 'CLAP_weights_2023.pth')
GPT2_LOCAL_PATH = os.path.join(LOCAL_MODEL_DIR, 'gpt2')
ROBERTA_LOCAL_PATH = release_path('${CLAP_SOURCE}/model/roberta-base')
import msclap.CLAPWrapper

def offline_hf_hub_download(*args, **kwargs):
    if not os.path.exists(CLAP_WEIGHTS_PATH):
        raise FileNotFoundError(f'Missing CLAP weights: {CLAP_WEIGHTS_PATH}')
    return CLAP_WEIGHTS_PATH
msclap.CLAPWrapper.hf_hub_download = offline_hf_hub_download
import transformers

def patch_transformers_offline(cls_name):
    cls = getattr(transformers, cls_name)
    orig_func = cls.from_pretrained

    @classmethod
    def my_func(cls_inner, pretrained_model_name_or_path, *args, **kwargs):
        target_path = GPT2_LOCAL_PATH if 'gpt2' in str(pretrained_model_name_or_path).lower() else ROBERTA_LOCAL_PATH
        kwargs['local_files_only'] = True
        return orig_func.__func__(cls_inner, target_path, *args, **kwargs)
    setattr(cls, 'from_pretrained', my_func)
for cls_name in ['AutoModel', 'AutoConfig', 'AutoTokenizer', 'GPT2Tokenizer', 'RobertaTokenizer']:
    try:
        patch_transformers_offline(cls_name)
    except AttributeError:
        continue
from msclap import CLAP
from pykeen.predict import predict_target
from pykeen.triples import TriplesFactory
ESC50_CSV = release_path('${DATA_ROOT}/ESC-50/esc50.csv')
ESC50_AUDIO_DIR = release_path('${DATA_ROOT}/ESC-50/audio')
KGE_MODEL_DIR = release_path('${KGE_ROOT}/001/TFVpYwo2_RotatE_False')
TRAIN_TRIPLES_PATH = os.path.join(KGE_MODEL_DIR, 'AKG_train_triples.tsv')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TOP_K = 5
TOP_M = 3
LOGIT_SCALE = 100.0
HOP1_RELATIONS = ['belongs to class', 'has parent', 'perceived as', 'event composed of', 'has children']

def to_tensor(emb):
    if isinstance(emb, torch.Tensor):
        return emb
    return torch.from_numpy(emb)

def get_safe_text_embeddings(model, text_list, device):
    if not text_list:
        return torch.empty((0,), device=device)
    embs = []
    for text in text_list:
        emb = model.get_text_embeddings([text])
        embs.append(to_tensor(emb).to(device).float())
    return torch.cat(embs, dim=0)

def score_prompt_list(clap_model, audio_embed, prompts):
    if not prompts:
        return torch.empty((0,), device=audio_embed.device)
    prompt_embs = F.normalize(get_safe_text_embeddings(clap_model, prompts, audio_embed.device), dim=-1)
    scores = torch.matmul(audio_embed, prompt_embs.T).squeeze()
    if scores.dim() == 0:
        scores = scores.unsqueeze(0)
    return scores

def rank_of_true(score_vec, true_indices):
    order = torch.argsort(score_vec, descending=True).detach().cpu().numpy()
    return min((int(np.where(order == target_idx)[0][0] + 1) for target_idx in true_indices))

def get_kg_entity(class_name):
    clean_name = class_name.replace('_', ' ').strip()
    if '(' in clean_name:
        clean_name = clean_name.split('(')[0].strip()
    return clean_name.replace(' - ', ' ')

def resolve_head_query(head, training_factory):
    return head

def build_tail_predictor(kge_model, training_factory):
    cache = {}

    def get_tails(head, rel):
        key = (head, rel)
        if key in cache:
            return cache[key]
        query_head = resolve_head_query(head, training_factory)
        if query_head not in training_factory.entity_to_id:
            cache[key] = []
            return []
        try:
            pred = predict_target(model=kge_model, head=query_head, relation=rel, triples_factory=training_factory)
            result = pred.df.sort_values(by='score', ascending=False).head(TOP_M)['tail_label'].tolist()
        except Exception:
            result = []
        cache[key] = result
        return result
    return get_tails

def load_dataset():
    df = pd.read_csv(ESC50_CSV)
    kg_classes = sorted(df['category'].unique())
    label_classes = [cat.replace('_', ' ') for cat in kg_classes]
    class_to_idx = {cat: idx for idx, cat in enumerate(kg_classes)}
    return {'df': df, 'label_classes': label_classes, 'kg_classes': kg_classes, 'class_labels_set': {name.lower() for name in label_classes}, 'class_to_idx': class_to_idx, 'total': len(df)}

def iter_samples(dataset):
    class_to_idx = dataset['class_to_idx']
    for _, row in dataset['df'].iterrows():
        true_indices = [class_to_idx[row['category']]] if row['category'] in class_to_idx else []
        yield {'audio_path': os.path.join(ESC50_AUDIO_DIR, row['filename']), 'true_indices': true_indices}
