from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from starlette.requests import Request
import logging
import warnings
warnings.simplefilter('ignore')

import os
import pickle
import pandas as pd
import scipy
from sentence_transformers import SentenceTransformer
# from transformers import pipeline

from starlette.middleware.cors import CORSMiddleware

BIORXIV_PATH = 'data/biorxiv_medrxiv/biorxiv_medrxiv/'
COMM_USE_PATH = 'data/comm_use_subset/comm_use_subset/'
NONCOMM_USE_PATH = 'data/noncomm_use_subset/noncomm_use_subset/'
METADATA_PATH = 'data/metadata.csv'

DATA_PATH = 'data'
MODELS_PATH = 'models'
MODEL_NAME = 'scibert-nli'
CORPUS_PATH = os.path.join(DATA_PATH, 'corpus.pkl')
MODEL_PATH = os.path.join(MODELS_PATH, MODEL_NAME)
EMBEDDINGS_PATH = os.path.join(DATA_PATH, f'{MODEL_NAME}-embeddings.pkl')

# summarizer = pipeline(task='summarization')
app = FastAPI()

origins = [
    "http://metromanilaquarantine.netlify.com",
    "https://metromanilaquarantine.netlify.com/",
    "http://159.65.136.34/",
    "http://159.65.136.34:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    question: str = ""
    abstracts:List[str] = []
    # summary: str = ""

def ask_question(query, model, corpus, corpus_embed, top_k=5):
    """
    Adapted from https://www.kaggle.com/dattaraj/risks-of-covid-19-ai-driven-q-a
    """
    queries = [query]
    query_embeds = model.encode(queries, show_progress_bar=False)
    for query, query_embed in zip(queries, query_embeds):
        distances = scipy.spatial.distance.cdist([query_embed], corpus_embed, "cosine")[0]
        distances = zip(range(len(distances)), distances)
        distances = sorted(distances, key=lambda x: x[1])
        results = []
        for count, (idx, distance) in enumerate(distances[0:top_k]):
            results.append([count + 1, corpus[idx].strip(), round(1 - distance, 4)])
    return results

# pickle already preloaded via interactive_search.py
with open(CORPUS_PATH, 'rb') as corpus_pt:
    corpus = pickle.load(corpus_pt)

model = SentenceTransformer(MODEL_PATH)

if not os.path.exists(EMBEDDINGS_PATH):
    print("Computing and caching model embeddings for future use...")
    embeddings = model.encode(corpus, show_progress_bar=True)
    with open(EMBEDDINGS_PATH, 'wb') as file:
        pickle.dump(embeddings, file)
else:
    print("Loading model embeddings from", EMBEDDINGS_PATH, '...')
    with open(EMBEDDINGS_PATH, 'rb') as file:
        embeddings = pickle.load(file)    

@app.get("/")
async def read_root():
    return {"welcome to manila covid api"}

@app.post("/chat")
async def post_chat(request: Request, item: Item):
    if request.method == "POST":
        item_dict = item.dict()
        abstracts = ask_question(item.question, model, corpus, embeddings)
        item_dict.update({"abstracts":abstracts})
        # all_abstracts = ''
        # for res in abstracts:
        #     text = res[1]
        #     text = textwrap.fill(text, width=75)
        #     all_abstracts = all_abstracts + ' ' + text        
        # summary = summarizer(all_abstracts, max_length=40)
        # item_dict.update({"summary":summary[0]['summary_text']})

    return item_dict

