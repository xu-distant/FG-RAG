import argparse


def parse_args_llama():
    parser = argparse.ArgumentParser(description="GRAG")

    parser.add_argument("--model_name", type=str, default='GRAG')
    parser.add_argument("--project", type=str, default="projection")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--dataset", type=str, default='hotpotqa')
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--patience", type=float, default=5)

    # Model Training
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_steps", type=int, default=2)

    # Learning Rate Scheduler
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--warmup_epochs", type=float, default=1)

    # Inference
    parser.add_argument("--eval_batch_size", type=int, default=16)

    # LLM related
    parser.add_argument("--llm_model_name", type=str, default='7b-hf')
    parser.add_argument("--llm_model_path", type=str, default='')
    parser.add_argument("--llm_frozen", type=str, default='True')
    parser.add_argument("--llm_num_virtual_tokens", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default='output')
    parser.add_argument("--max_txt_len", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=150)

    # GraphLLM related
    parser.add_argument("--ddm_input_dim", type=int, default=4096)
    parser.add_argument("--ddm_hidden_dim", type=int, default=1024)
    parser.add_argument("--ddm_latent_dim", type=int, default=1024)
    parser.add_argument("--ddm_max_length", type=int, default=15)
    parser.add_argument("--ddm_num_timesteps", type=int, default=1000)
    parser.add_argument("--ddm_num_infer_steps", type=int, default=200)
    # GNN related
    parser.add_argument("--gnn_model_name", type=str, default='gat')
    parser.add_argument("--gnn_num_layers", type=int, default=4)
    parser.add_argument("--gnn_in_dim", type=int, default=1024)
    parser.add_argument("--gnn_hidden_dim", type=int, default=1024)
    parser.add_argument("--alignment_mlp_layers", type=int, default=3)
    parser.add_argument("--gnn_num_heads", type=int, default=4)
    parser.add_argument("--distance_operator", type=str, default='euclidean')
    parser.add_argument("--gnn_dropout", type=float, default=0.0)

    # RL related
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps_clip", type=float, default=0.2)
    parser.add_argument("--K_epochs", type=int, default=4)
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument("--n", type=int, default=5)
    # PPO specific
    parser.add_argument("--ppo_lr", type=float, default=0.001)
    parser.add_argument("--ppo_hidden_dim", type=int, default=512)
    parser.add_argument("--ppo_m", type=int, default=10)
    parser.add_argument("--ppo_n", type=int, default=5)
    parser.add_argument("--gae_lambda", type=float, default=0.9)
    args = parser.parse_args()
    return args
