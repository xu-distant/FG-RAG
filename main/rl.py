import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import random
from utils.collate import collate_fn
from collections import deque
from utils.config import parse_args_llama
from GraphLLM_test import GraphLLM
from utils.ckpt import _reload_best_model
from torch.utils.data import DataLoader



def save_model(ppo_model, checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = 'ppo_model.pth'
    checkpoint = {
        'model_state_dict': ppo_model.state_dict(), }
    torch.save(checkpoint, checkpoint_path)
    print(f"Model saved to: {checkpoint_path}")
def load_model(ppo_model, checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = 'ppo_model.pth'
    checkpoint = torch.load(checkpoint_path)
    ppo_model.load_state_dict(checkpoint["model_state_dict"])

    return ppo_model
class Memory:
    """Experience replay buffer, storing the trajectory during training"""
    def __init__(self):
        self.actions = []
        self.states = []
        self.available_mask=[]
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.state_values = [] 
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.available_mask[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.state_values[:]

class ActorCritic(nn.Module):
    """Actor-Critic network"""
    def __init__(self, embedding_dim, hidden_dim):
        super(ActorCritic, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.actor = nn.Sequential(
            nn.MultiheadAttention(embedding_dim, num_heads=4),
            nn.LSTM(embedding_dim, hidden_dim, batch_first=True),
            nn.Linear(hidden_dim, 1),  
        ).to(self.device)
        self.critic = nn.Sequential(
            nn.MultiheadAttention(embedding_dim, num_heads=4),
            nn.LSTM(embedding_dim, hidden_dim, batch_first=True),
            nn.Linear(hidden_dim, 1)  
        ).to(self.device)

    def forward(self, embeddings, available_mask=None):
             attn_output, _ = self.actor[0](embeddings, embeddings, embeddings)
             lstm_out, _ = self.actor[1](attn_output)
             action_probs = self.actor[2](lstm_out).squeeze(-1)

             if available_mask is not None:
                 action_probs = action_probs.masked_fill(available_mask == 0, float('-inf'))
             action_probs = torch.softmax(action_probs, dim=-1)
             if available_mask is not None:
                 embeddings = embeddings[available_mask == 1]  
             attn_output, _ = self.critic[0](embeddings, embeddings, embeddings)
             lstm_out, _ = self.critic[1](attn_output)
             state_value = self.critic[2](lstm_out)

             return action_probs, state_value.mean(dim=0)

class PPO:
    """PPO algorithm implementation with Generalized Advantage Estimation (GAE)"""
    def __init__(self, embedding_dim, hidden_dim, m, n, alpha, lr=0.001,
                 gamma=0.99, eps_clip=0.2, K_epochs=4, gae_lambda=0.95):
        self.m = m  
        self.n = n  
        self.alpha = alpha  
        self.gamma = gamma  
        self.eps_clip = eps_clip  # PPO clipping parameter
        self.K_epochs = K_epochs  
        self.gae_lambda = gae_lambda  # GAE lambda parameter: controls bias-variance tradeoff
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.policy = ActorCritic(embedding_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(embedding_dim, hidden_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
       
        self.MseLoss = nn.MSELoss()
        self.memory = Memory()
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def select_action(self, embeddings, available_mask):    
        with torch.no_grad():
            action_probs, state_value = self.policy_old(embeddings, available_mask)
            
        dist = Categorical(action_probs)
        action = dist.sample()
        self.memory.states.append(embeddings)
        self.memory.available_mask.append(available_mask.clone())
        self.memory.actions.append(action)
        self.memory.logprobs.append(dist.log_prob(action))
        self.memory.state_values.append(state_value.detach()) 
        
        return action.item()

    def get_reward(self, selected_embeddings, question_emb, standard_answer, question,model):
        """Calculate reward"""
        # R1: LLM evaluation reward.
        concat_emb = concat_embeddings(selected_embeddings)
        llm_answer = model.inference1(concat_emb,question)
        r1 =calculate_answer_similarity(llm_answer, standard_answer)
        # R2: Embedding similarity reward.
        r2 = calculate_embedding_similarity(selected_embeddings, question_emb)
        return self.alpha * r1 + (1 - self.alpha) * r2


    def compute_gae(self, rewards, state_values, is_terminals):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0 if is_terminals[t] else state_values[t]
            else:
                next_value = state_values[t + 1]
            delta = rewards[t] + self.gamma * next_value - state_values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
        advantages = torch.tensor(advantages).to(self.device)
        return advantages


    def update(self):
        
        rewards = torch.tensor(self.memory.rewards).to(self.device)
        state_values = torch.stack(self.memory.state_values).squeeze(-1).detach()
        is_terminals = self.memory.is_terminals
        advantages = self.compute_gae(rewards, state_values, is_terminals)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + state_values.detach()
        old_states = torch.stack(self.memory.states).detach()
        old_actions = torch.stack(self.memory.actions).detach()
        old_logprobs = torch.stack(self.memory.logprobs).detach()
        old_available_mask = torch.stack(self.memory.available_mask).detach()
    
        for _ in range(self.K_epochs):
            action_probs_list = []
            state_values_list = []

            for i in range(len(old_available_mask)):
                action_probs, state_values_new = self.policy(old_states[i], old_available_mask[i])
                action_probs_list.append(action_probs)
                state_values_list.append(state_values_new)

            action_probs = torch.stack(action_probs_list)
            state_values_new = torch.stack(state_values_list).squeeze(-1)
            
            dist = Categorical(action_probs)
            new_logprobs = dist.log_prob(old_actions)
            
            ratios = torch.exp(new_logprobs - old_logprobs.detach())
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2)
            critic_loss = self.MseLoss(state_values_new, returns)
            loss = actor_loss + 0.5 * critic_loss
            loss = loss.mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.memory.clear_memory()

def concat_embeddings( embeddings):
    size, num_nodes, hidden_dim = embeddings.shape
    return embeddings.reshape(-1, hidden_dim)

def get_llm_response(self, concat_emb):
    pass

def calculate_answer_similarity(llm_answer, standard_answer):

     exact_match = 1.0 if llm_answer == standard_answer else 0.0
     return exact_match

def calculate_embedding_similarity(selected_embeddings, question_emb):
    similarities = []
    question_emb=question_emb.squeeze()
    question_emb=question_emb.mean(dim=0)
    for emb in selected_embeddings:
        emb=emb.mean(dim=0)
        sim = torch.cosine_similarity(emb, question_emb, dim=0)
        similarities.append(sim)
    return torch.mean(torch.stack(similarities))

def train(ppo_agent, train_loader,val_loader, args):
    model = GraphLLM(args)
    model = _reload_best_model(model, args)
    model.eval()
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_val_reward = float('-inf')
    best_epoch = 0
    for epoch in range(args.num_epochs):
        ppo_agent.policy.train()
        for step, batch in enumerate(train_loader):
            graphs = batch['sub_graphs'][0]
            graphs_embeds = []
            for graph in graphs:
                x=graph.node_embeddings.to(device)
                edge_index=graph.edge_index.to(device)
                with torch.no_grad():
                    graph_embed = model.graph_encoder.infer(x,edge_index,150,device)
                graphs_embeds.append(graph_embed)
            graph_embeds = torch.stack(graphs_embeds).to(device)
            selected_indices = []
            available_mask = torch.ones(ppo_agent.m).to(device)

            for _ in range(ppo_agent.n):
                action = ppo_agent.select_action(graph_embeds.mean(dim=1), available_mask)
                selected_indices.append(action)
                available_mask[action] = 0

            selected_embeddings = graph_embeds[selected_indices]
            reward = ppo_agent.get_reward(selected_embeddings, batch['question_embeddings'][0], batch['answer'][0],
                                     batch['question'][0], model)

            for _ in range(ppo_agent.n):
                ppo_agent.memory.rewards.append(reward)
                ppo_agent.memory.is_terminals.append(False)
            ppo_agent.memory.is_terminals[-1] = True
            if len(ppo_agent.memory.states) >= 64: 
                print(step)
                ppo_agent.update()
        print(f"Epoch {epoch + 1} completed")
        ppo_agent.memory.clear_memory()
        cnt=0
        siml=0.
        ppo_agent.policy.eval()
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                graphs = batch['sub_graphs'][0]
                graphs_embeds = []
                for graph in graphs:
                    x = graph.node_embeddings.to(device)
                    edge_index = graph.edge_index.to(device)
                    graph_embed = model.graph_encoder.infer(x, edge_index, args.ddm_num_infer_steps, device)
                    graphs_embeds.append(graph_embed)
                graph_embeds = torch.stack(graphs_embeds).to(device)
                selected_indices = []
                available_mask = torch.ones(ppo_agent.m).to(device)
                for _ in range(ppo_agent.n):
                    action = ppo_agent.select_action(graph_embeds.mean(dim=1), available_mask)
                    selected_indices.append(action)
                    available_mask[action] = 0

                selected_embeddings = graph_embeds[selected_indices]
                cnt+=1
                siml+=calculate_embedding_similarity(selected_embeddings, batch['question_embeddings'][0])
                
            avg_reward = siml / cnt  
            print(siml/cnt)
        if avg_reward > best_val_reward:
            best_val_reward = avg_reward
            save_model(model)
            best_epoch = epoch
        print(f'Epoch {epoch + 1} avg_reward {avg_reward} Best Val reward {best_val_reward} Best Epoch {best_epoch}')

        if epoch - best_epoch >= args.patience:
            print(f'Early stop at epoch {epoch + 1}')
            break
args=parse_args_llama()
batch_size = args.batch_size 
subgraphs_train = torch.load('/data/processed/hotpotqa_train.pt')
train_loader = DataLoader(subgraphs_train, batch_size=batch_size, drop_last=True, shuffle=True, collate_fn=collate_fn)
batch_size = args.batch_size  
subgraphs_val = torch.load('/data/processed/hotpotqa_val.pt')
val_loader = DataLoader(subgraphs_val, batch_size=batch_size, drop_last=True, shuffle=True, collate_fn=collate_fn)

ppo_agent = PPO(
    embedding_dim=args.ddm_input_dim,
    hidden_dim=args.ppo_hidden_dim,
    m=args.ppo_m,
    n=args.ppo_n,
    alpha=args.alpha,
    lr=args.ppo_lr,
    gamma=args.gamma,
    eps_clip=args.eps_clip,
    K_epochs=args.K_epochs,
    gae_lambda=args.gae_lambda
)

train(ppo_agent, train_loader, val_loader, args)
# Evaluate model
# # Clean CUDA cache
# torch.cuda.empty_cache()
# torch.cuda.reset_max_memory_allocated()
# args=parse_args_llama()
# model=GraphLLM(args)
# model = _reload_best_model(model, args)
# model.eval()
# ppo_agent = load_model(model)
# ppo_agent.policy.eval()
# batch_size = args.batch_size
# subgraphs_test=torch.load('/data/processed/hotpotqa_test.pt')
# test_loader = DataLoader(subgraphs_test, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# eval_output = []
# true_answers = []
# similarity=[]
# with torch.no_grad():
#     for step, batch in enumerate(test_loader):
#         graphs = batch['sub_graphs'][0]
#         for graph in graphs:
#             x = graph.node_embeddings.to(device)
#             edge_index = graph.edge_index.to(device)
#             graph_embed = model.graph_encoder.infer(x, edge_index, 150, device)
#             graphs_embeds.append(graph_embed)
#         graph_embeds = torch.stack(graphs_embeds).to(device)
#         selected_indices = []
#         available_mask = torch.ones(ppo_agent.m).to(device)
#         for _ in range(ppo_agent.n):
#             with torch.no_grad():
#                 action = ppo_agent.select_action(graph_embeds.mean(dim=1), available_mask)
#             selected_indices.append(action)
#             available_mask[action] = 0
#         selected_embeddings = graph_embeds[selected_indices]
#         concat_emb = concat_embeddings(selected_embeddings)
#         llm_answer = model.inference1(concat_emb, batch['question'][0])
#         output = model.inference1(batch)
#         eval_output.append(output)
#         true_answers.append(batch['answer'][0])
#         sim=calculate_embedding_similarity(selected_embeddings, batch['question_embeddings'][0])
#         similarity.append(sim)
# predictions = []  
# true_labels = [] 
#
# # Calculate EM (Exact Match)
# def exact_match_score(predictions, true_labels):
#     return sum([1 if pred == true else 0 for pred, true in zip(predictions, true_labels)]) / len(true_labels)
#
# for i, output in enumerate(eval_output):
#     predicted_answer = output['pred']  
#     true_answer = true_answers[i]  
#     predictions.append(predicted_answer)  
#     true_labels.append(true_answer)  
#
# em_score = exact_match_score(predictions, true_labels)  
# print(f"Exact Match Score: {em_score}")  
# sim_between_context_and_question=similarity.stack().mean(dim=0)
# print(f"sim_between_context_and_question: {sim_between_context_and_question}")