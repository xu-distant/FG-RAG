import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from torch.nn.functional import cosine_similarity
from torch_geometric.data import Data
import networkx as nx
from keyword_match import keywords_match, keywords_match_quik
from article_to_sentence import generate_propositions,generate_propositions_batch
from transformers import LlamaTokenizer, LlamaModel
# Load data
def load_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
def preprocess_subgraphs(file_path,m,lens,thred_score,dim):
    data = load_data(file_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = LlamaTokenizer.from_pretrained("/data/model/Llama-2-7b-hf")
    model = LlamaModel.from_pretrained("/data/model/Llama-2-7b-hf").to(device)
    tokenizer_propositions = AutoTokenizer.from_pretrained("/data/model/propositionizer-wiki-flan-t5-large")
    model_propositions = AutoModelForSeq2SeqLM.from_pretrained("/data/model/propositionizer-wiki-flan-t5-large").to(device)
    # Store all subgraphs
    all_subgraphs = []
    for index,row in enumerate(data):
        if len(all_subgraphs) >= lens: break
        question = row['question']
        context = row['context']
        G = nx.Graph()
        sentences = []
        for title, paragraph in context:
            sentences.append(paragraph)
        sentences = generate_propositions_batch(sentences, model_propositions, tokenizer_propositions, device)
        for sentence in sentences:
            score = keywords_match_quik(sentence, question)
            if score > thred_score: G.add_node(sentence)
        if len(G.nodes)==0 : continue
        edge_attributes = []
        nodes_list = list(G.nodes)
        for i, sentence in enumerate(nodes_list):
            for other_sentence in nodes_list[i + 1:]:
                score = keywords_match_quik(sentence, other_sentence)
                if score > thred_score:
                    G.add_edge(sentence, other_sentence)
                    edge_attributes.append((sentence, other_sentence, score))
        embeddings  = model.encode(sentences, convert_to_tensor=True, device=device).last_hidden_state
        print(embeddings.shape)
        node_embeddings = torch.zeros((len(G.nodes), dim), device=device)
        for idx, node in enumerate(nodes_list):
            neighbors = list(G.neighbors(node))
            if neighbors:
                node_embedding = embeddings[idx]
                neighbor_indices = [list(G.nodes).index(neighbor) for neighbor in neighbors]
                neighbor_embeddings = torch.mean(embeddings[neighbor_indices], dim=0)
                node_embeddings[idx] = torch.mean(torch.stack([node_embedding, neighbor_embeddings]), dim=0)
            else:
                node_embeddings[idx] = embeddings[idx]

        question_embedding = model.encode([question], convert_to_tensor=True, device=device).last_hidden_state
        similarities = cosine_similarity(
            question_embedding.unsqueeze(1),
            node_embeddings.unsqueeze(0),
            dim=2)[0]
        sorted_nodes = sorted(enumerate(similarities.cpu().numpy()), key=lambda item: item[1], reverse=True)
        top_m_nodes = [list(G.nodes)[node] for node, _ in sorted_nodes[:m]]
        sub_graphs = []
        for node in top_m_nodes:
            subgraph_nodes = [node] + list(G.neighbors(node))
            subgraph = G.subgraph(subgraph_nodes)
            edges = list(subgraph.edges)
            unique_nodes = list(subgraph.nodes)
            edge_mapping = {node: idx for idx, node in enumerate(unique_nodes)}  # Create a mapping from nodes to new indices
            edge_index = torch.tensor([[edge_mapping[edge[0]], edge_mapping[edge[1]]] for edge in edges],
                                      dtype=torch.long).t().contiguous()
            node_indices = [list(G.nodes).index(n) for n in subgraph.nodes]
            x = torch.stack([embeddings[idx] for idx in node_indices])
            data=Data(edge_index=edge_index)
            data.node_embeddings = x
            sub_graphs.append(data)

        all_subgraphs.append({
            'answer': row['answer'],
            'question_embeddings': question_embedding,
            'id': row['_id'],
            'question': row['question'],
            'sub_graphs': sub_graphs
        })
    return all_subgraphs


