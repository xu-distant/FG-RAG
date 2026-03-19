import re
import string
from collections import Counter
import torch
import wandb
from datasets import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from GraphLLM_test import GraphLLM
from utils.collate import collate_fn
from utils.config import parse_args_llama
from utils.lr_schedule import adjust_learning_rate
from utils.ckpt import _save_checkpoint, _reload_best_model
from utils.seed import seed_everything

args = parse_args_llama()
#Step 1: Set up wandb
seed = args.seed
# wandb.init(project="RAG",
#            name=f"{args.dataset}_{args.model_name}_seed{seed}",
#            config=args)
seed_everything(seed=args.seed)
#
# #step 2. Prepare data loader
print('Start training!')
batch_size = args.batch_size
subgraphs_train=torch.load('/data/processed/hotpotqa_train.pt')
train_loader = DataLoader(subgraphs_train, batch_size=batch_size, drop_last=True, shuffle=True, collate_fn=collate_fn)
subgraphs_val=torch.load('/data/processed/hoypotqa_val.pt')
val_loader = DataLoader(subgraphs_val, batch_size=batch_size, drop_last=True, shuffle=True, collate_fn=collate_fn)

model = GraphLLM(args=args)
# Step 3 Set Optimizer
params = [p for _, p in model.named_parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(
    [{'params': params, 'lr': args.lr, 'weight_decay': args.wd}, ],
    betas=(0.9, 0.95)
)
trainable_params, all_param = model.print_trainable_params()
print(
    f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

    # step 4. Train the model
num_training_steps = args.num_epochs * len(train_loader)
progress_bar = tqdm(range(num_training_steps))
best_val_loss = float('inf')
for epoch in range(args.num_epochs):
    model.train()
    epoch_loss, accum_loss = 0., 0.
    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        clip_grad_norm_(model.parameters(), 0.1)
        if (step + 1) % args.grad_steps == 0:
            adjust_learning_rate(optimizer.param_groups[0],args.lr, step / len(train_loader) + epoch, args)
        optimizer.step()
        epoch_loss, accum_loss = epoch_loss + loss.item(), accum_loss + loss.item()
        if (step + 1) % args.grad_steps == 0:
            lr = optimizer.param_groups[0]["lr"]
            accum_loss = 0.
        progress_bar.update(1)
    print(f"Epoch: {epoch + 1}/{args.num_epochs}: Train Loss (Epoch Mean): {epoch_loss / len(train_loader)}")
    val_loss = 0.
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            loss = model(batch)
            val_loss += loss.item()
        val_loss = val_loss / len(val_loader)
        print(f"Epoch: {epoch + 1}/{args.num_epochs}: Val Loss: {val_loss}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        _save_checkpoint(model, optimizer, epoch, args, is_best=True)
        best_epoch = epoch

    print(f'Epoch {epoch + 1} Val Loss {val_loss} Best Val Loss {best_val_loss} Best Epoch {best_epoch}')

    if epoch - best_epoch >= args.patience:
        print(f'Early stop at epoch {epoch + 1}')
        break

    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

model=GraphLLM(args)
model = _reload_best_model(model, args)
model.eval()

subgraphs_test=torch.load('/data/processed/hotpotqa_test.pt')
test_loader = DataLoader(subgraphs_test, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
eval_output = []
true_answers = []
progress_bar_test = tqdm(range(len(test_loader)))

for step, batch in enumerate(test_loader):
    with torch.no_grad():
        output = model.inference(batch)
        eval_output.append(output)
        true_answers.append(batch['answer'])
    progress_bar_test.update(1)

# Step 6. Post-processing & compute metrics
predictions = []
true_labels = []

# Calculate EM (Exact Match)
def exact_match_score(predictions, true_labels):
    return sum([1 if pred == true else 0 for pred, true in zip(predictions, true_labels)]) / len(true_labels)

for i, output in enumerate(eval_output):
    predicted_answer = output['pred']
    true_answer = true_answers[i]
    predictions.append(predicted_answer)
    true_labels.append(true_answer)

# Calculate EM value
em_score = exact_match_score(predictions, true_labels)
print(f"Exact Match Score: {em_score}")
# Calculate F1 score
def normalize_text(text: str) -> str:

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(lower(text)))

def calc_f1(pred: str, answer: str) -> float:
    norm_pred = normalize_text(pred).split()
    norm_answer = normalize_text(answer).split()
    common_tokens = Counter(norm_pred) & Counter(norm_answer)
    num_same = sum(common_tokens.values())

    if len(norm_pred) == 0 or len(norm_answer) == 0:
        return 0.0  # Avoid division by zero

    precision = num_same / len(norm_pred)
    recall = num_same / len(norm_answer)

    if precision + recall == 0:
        return 0.0  # Avoid division by zero

    f1 = 2 * precision * recall / (precision + recall)
    return f1

scores = []
for text, answers in zip(predictions, true_labels):
    scores.append(calc_f1(text[0], answers[0]))
print(sum(scores) / len(scores))