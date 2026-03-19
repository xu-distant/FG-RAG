import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5ForConditionalGeneration
import faiss
def get_docs(query):
    wikipedia_data = pd.read_csv('/data/retrieval_wiki/psgs_w100.tsv', sep='\t')

    embeddings_dir = '/data/retrieval_wiki/wikipedia_embeddings'

    embeddings = []
    for file_name in os.listdir(embeddings_dir):
        file_path = os.path.join(embeddings_dir, file_name)
        embedding = np.load(file_path)
        embeddings.append(embedding)
    embeddings = np.vstack(embeddings).astype('float32')  

    dimension = embeddings.shape[1] 
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Contriever and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/data/model/contriever-msmarco")
    model = AutoModel.from_pretrained("/data/model/contriever-msmarco")

    query = "query"
    query_inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        query_embedding = model(**query_inputs).last_hidden_state.mean(dim=1).numpy().astype('float32')

    # FAISS retrieval
    k = 10  
    D, I = index.search(query_embedding, k=k)

    # Get the most relevant documents
    most_similar_docs = wikipedia_data.iloc[I[0]]
    print("The most relevant documents:")
    for idx, doc in most_similar_docs.iterrows():
        print(f"文档 {idx}: {doc['text']}")



