import torch
import os
import torch.nn as nn
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv, global_mean_pool
from utils.collate import collate_fn
from torch.utils.data import DataLoader
from utils.config import parse_args_llama
class ModelCheckpoint:
    def __init__(self):
        self.save_dir = '/data/GRAG/output/DDM'
        os.makedirs(self.save_dir, exist_ok=True)
    
    def save_model(self, model, optimizer, epoch, loss, filename=None):
        """
        Save the model, optimizer state, and training information.
        """
        if filename is None:
            filename = f'DDM_model.pth'
            
        checkpoint_path = os.path.join(self.save_dir, filename)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"The model has been saved to: {checkpoint_path}")
    
    def load_model(self, model, optimizer, checkpoint_path):
        """
        加载模型和优化器状态
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"找不到检查点文件: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path)
        
        # Load the model and optimizer state
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint['epoch'], checkpoint['loss']

class GNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class DDMGraphModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim,max_length=15, num_timesteps=500):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.max_length=max_length
        # time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        # Encoder (the first two GNN layers)
        self.encoder_gnns = nn.ModuleList([
            GNNLayer(input_dim*2, hidden_dim),
            GNNLayer(hidden_dim, hidden_dim)
        ])
        
        # Decoder (the last two GNN layers)
        self.decoder_gnns = nn.ModuleList([
            GNNLayer(hidden_dim, hidden_dim),
            GNNLayer(hidden_dim, latent_dim)
        ])
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim)
        )

        self.readout_attention = nn.Sequential(
           nn.Linear(input_dim, 1)
       )

        self.sentence_mapping = nn.Sequential(
           nn.Linear(input_dim, max_length * input_dim),
           nn.ReLU(),
           nn.Dropout(0.1)
       )
         # Noise schedule
        self.register_buffer('betas', self._get_beta_schedule())
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alpha_cumprod', torch.sqrt(1 - alphas_cumprod))
        
    def _get_beta_schedule(self):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, self.num_timesteps)
    def _get_time_embedding(self, t):
        t = t.reshape(-1, 1).float()
        return self.time_embed(t)
    def graph_readout(self, node_features):
        graph_repr=node_features.mean(dim=0)
        sentence_features = self.sentence_mapping(graph_repr)  # [max_length * input_dim]
        sentence_features = sentence_features.reshape(self.max_length, -1)  # [max_length, input_dim]
        return sentence_features
    
    def forward(self, x, edge_index, t):
        # Get time embedding
        t_emb = self._get_time_embedding(t)
        # Concatenate the time information with the node features.
        x = torch.cat([x, t_emb], dim=-1)
        # Encoder stage (denoising process)
        if edge_index.size(0)==0:
            h=self.fc(x)
        else:
            h = x
            skip_connection = None

            for i, encoder_gnn in enumerate(self.encoder_gnns):
                h = encoder_gnn(h, edge_index)
                if i == 0:
                    skip_connection = h
            h = h + skip_connection
            # Decoder stage
            for decoder_gnn in self.decoder_gnns:
                h = decoder_gnn(h, edge_index)

        latent_code = self.mlp(h)
        return latent_code

    def add_noise(self, x, t):
       epsilon = torch.randn_like(x)
       if x.shape[0] == 1:
           bar_epsilon= epsilon
       else:
           mu = x.mean(dim=0, keepdim=True)
           sigma = x.std(dim=0, keepdim=True)
           bar_epsilon = mu + sigma * epsilon  # \bar{\epsilon} = \mu + \sigma \odot \epsilon
       
       # Ensure noise shares the same sign as the features
       epsilon_prime = torch.sign(x) * torch.abs(bar_epsilon)  # \epsilon' = \text{sgn}(x_{0,i}) \odot |\bar{\epsilon}|
       
       # Compute the noise-added features.
       sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t].view(-1, 1)
       sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1)
       
       return sqrt_alpha_cumprod * x + sqrt_one_minus_alpha_cumprod * epsilon_prime
    def infer(self, x, edge_index, num_steps,device):
       """
       Inference stage: gradually denoising starting from the noisy graph data.
       """
       x=x.mean(dim=1)
       x_t = x
       for t in reversed(range(num_steps)):
           sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t].view(-1, 1)
           sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1)
           t = torch.full((x_t.size(0),), t, device=device)
           predicted_x0 = self.forward(x_t, edge_index, t)
           
           noise = torch.randn_like(x_t) if t[0] > 0 else 0
           
           # Reverse diffusion
           x_t = sqrt_alpha_cumprod * predicted_x0 + sqrt_one_minus_alpha_cumprod * noise
       
       return self.graph_readout(x_t)
class DDMLoss(nn.Module):
    def __init__(self, lambda_smooth=0.1):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        
    def forward(self, pred, target, edge_index):
        # Reconstruction loss
        recon_loss = F.mse_loss(pred, target)
        return  recon_loss
       
def train_ddm(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in train_loader:

        optimizer.zero_grad()
        subgraph=batch['sub_graphs'][0][0]
        subgraph=subgraph.to(device)
        x=subgraph.node_embeddings
        edge_index=subgraph.edge_index
        if edge_index.size(0)==0:continue
        x=x.mean(dim=1)
    
        t = torch.randint(0, model.num_timesteps, (x.size(0),), device=device)
        noisy_features=model.add_noise(x,t)

        latent_code = model(noisy_features, edge_index, t)
        loss = criterion(latent_code, x,edge_index)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)




def main(args):
    input_dim = args.ddm_input_dim   
    hidden_dim = args.ddm_hidden_dim
    latent_dim = args.ddm_latent_dim
    max_length = args.ddm_max_length
    num_timesteps = args.ddm_num_timesteps
    batch_size = args.batch_size
    subgraphs_train=torch.load('/data/processed/hotpotqa_train.pt')
    train_loader = DataLoader(subgraphs_train, batch_size=batch_size, drop_last=True, shuffle=True, collate_fn=collate_fn)
    # Initialize the model.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DDMGraphModel(input_dim, hidden_dim, latent_dim,max_length,num_timesteps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = DDMLoss(lambda_smooth=0.1)
    # Create a checkpoint manager
    checkpoint_manager = ModelCheckpoint()
    # Training loop.
    best_loss = float('inf')
    num_epochs = args.num_epochs
    for epoch in range(num_epochs):
        train_loss = train_ddm(model, train_loader, optimizer, criterion, device)
        if train_loss < best_loss:
            best_loss = train_loss
            checkpoint_manager.save_model(model, optimizer, epoch, train_loss)
        print(f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f},best_loss: {best_loss:.4f}')

if __name__ == '__main__':
    args = parse_args_llama()
    main(args)