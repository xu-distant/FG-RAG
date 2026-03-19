from torch_geometric.data import Batch

def collate_fn_strategyqa(original_batch):
    batch = {}
    for k in original_batch[0].keys():
        batch[k] = [d[k] for d in original_batch]
    if 'graph' in batch:
         batch['graph'] = batch['graph'][0]
    if 'answer' in batch:
        batch['answer'] =  [str(item) for item in batch['answer']]
    return batch
