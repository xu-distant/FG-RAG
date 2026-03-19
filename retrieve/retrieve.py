import os
import time
import glob

import torch

from articles_retrieve.passage_retrieval import embed_queries, index_encoded_data, add_embeddings, validate, add_passages, add_hasanswer, load_data
from articles_retrieve.contriever_config import c_args
from articles_retrieve import src
device='cuda' if torch.cuda.is_available() else 'cpu'
def load_contriever():
    print(f"Loading model from: {c_args.model_name_or_path}")
    model, tokenizer, _ = src.contriever.load_retriever(c_args.model_name_or_path)
    return model, tokenizer

    # model.eval()
    # model = model.cuda()
    # if not c_args.no_fp16:
    #     model = model.half()

    # sentences = [
    #     "Where was Marie Curie born?",
    #     "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
    #     "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace."
    # ]

    # # Apply tokenizer
    # inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    # # Compute token embeddings
    # outputs = model(**inputs)
    # score1 = outputs[0] @ outputs[1]
    # score2 = outputs[0] @ outputs[2]
    # print(score1)
    # print(score2)
def load_passages_id_map():
    index = src.index.Indexer(c_args.projection_size, c_args.n_subquantizers, c_args.n_bits)
    # index all passages
    input_paths = glob.glob(c_args.passages_embeddings)
    input_paths = sorted(input_paths)
    embeddings_dir = os.path.dirname(input_paths[0])
    index_path = os.path.join(embeddings_dir, "index.faiss")
    if c_args.save_or_load_index and os.path.exists(index_path):
        index.deserialize_from(embeddings_dir)
    else:
        print(f"Indexing passages from files {input_paths}")
        start_time_indexing = time.time()
        index_encoded_data(index, input_paths, c_args.indexing_batch_size)
        print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
        if c_args.save_or_load_index:
            index.serialize(embeddings_dir)

    # load passages
    passages = src.data.load_passages(c_args.passages)
    passage_id_map = {x["id"]: x for x in passages}
    return passage_id_map, index

def beam_retrieve(input, contriever_model, contriever_tokenizer, passage_id_map, index):
    queries=[input]
    #queries = [input[0] + input[1]]
    # 要＋【】
    # queries = [ex["question"] for ex in data]
    questions_embedding = embed_queries(c_args, queries, contriever_model, contriever_tokenizer)
    # get top k results
    # start_time_retrieval = time.time()
    top_ids_and_scores = index.search_knn(questions_embedding, c_args.n_docs)
    # print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")
    m_docs = list()
    m_scores = list()
    for i, score in enumerate(top_ids_and_scores):
        docs = [passage_id_map[doc_id] for doc_id in score[0]]
        scores = [str(score) for score in score[1]]
        m_docs.append(docs)
        m_scores.append(scores)
        return m_docs, m_scores
    return m_docs
def retrieve(input):
    contriever, contriever_tokenizer = load_contriever()
    contriever=contriever.to(device)
    passage_id_map, index = load_passages_id_map()
    m_docs,m_scores = beam_retrieve(input, contriever, contriever_tokenizer, passage_id_map, index)
    return m_docs,m_scores
