import contextlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from DDM import DDMGraphModel

BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '</s>'
IGNORE_INDEX = -100
class GraphLLM(torch.nn.Module):
    def __init__(
        self,
        args,
        **kwargs
    ):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens
        print('Loading LLAMA')
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        kwargs = {
            "max_memory": {0: '20GiB', 1: '20GiB', 2: '20GiB', 3: '20GiB'},
            "device_map": device,
            "revision": "main",
        }

        self.tokenizer = AutoTokenizer.from_pretrained('/data/model/Llama-2-7b-hf', use_fast=False, revision=kwargs["revision"])
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'

        model = AutoModelForCausalLM.from_pretrained(
            '/data/model/Llama-2-7b-hf',
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            **kwargs
        )


        print("Freezing LLAMA!")
        for _, param in model.named_parameters():
            param.requires_grad = False

        self.model = model
        print('Finish loading LLAMA!')
        #DDM
        ddm_input_dim = args.ddm_input_dim
        ddm_hidden_dim = args.ddm_hidden_dim
        ddm_latent_dim = args.ddm_latent_dim
        ddm_max_length = args.ddm_max_length
        ddm_num_timesteps = args.ddm_num_timesteps

        DDMmodel = DDMGraphModel(
            input_dim=ddm_input_dim,
            hidden_dim=ddm_hidden_dim,
            latent_dim=ddm_latent_dim,
            max_length=ddm_max_length,
            num_timesteps=ddm_num_timesteps,
        ).to(device)
        checkpoint = torch.load('/data/GRAG/output/DDM/DDM_model.pth')
        DDMmodel.load_state_dict(checkpoint['model_state_dict'])
        self.graph_encoder = DDMmodel.to(device)
        self.word_embedding = self.model.model.get_input_embeddings()

    @property
    def device(self):
        return list(self.parameters())[0].device
    def maybe_autocast(self, dtype=torch.bfloat16):

        enable_autocast = self.device != torch.device("cpu")
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()


    def forward(self, samples):
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        labels = self.tokenizer(samples["answer"], add_special_tokens=False)
        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(
            self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.model.device)).unsqueeze(0)
        nodes=samples['sub_graphs'][0][0].node_embeddings.to(self.model.device)
        edge_index=samples['sub_graphs'][0][0].edge_index.to(self.model.device)

        graph_embeds = self.graph_encoder.infer(nodes,edge_index,90,self.model.device)

        batch_size =1
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        for i in range(batch_size):
    
            label_input_ids = labels.input_ids[i] + eos_tokens.input_ids
            input_ids = questions.input_ids[i] + eos_user_tokens.input_ids + label_input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))

            inputs_embeds = torch.cat([bos_embeds, graph_embeds, inputs_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [IGNORE_INDEX] * (inputs_embeds.shape[0] - len(label_input_ids)) + label_input_ids
            batch_label_input_ids.append(label_input_ids)

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )
    
        return outputs.loss

    def inference(self, samples):
        nodes = samples['sub_graphs'][0][0].node_embeddings.to(self.model.device)
        edge_index = samples['sub_graphs'][0][0].edge_index.to(self.model.device)
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(
            self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        graph_embeds = self.graph_encoder.infer(nodes, edge_index, 120, self.model.device)
        batch_size = 1
        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
        
            input_ids = questions.input_ids[i] + eos_user_tokens.input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([bos_embeds, graph_embeds, inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                # do_sample=True,
                use_cache=True,  

            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print('answer', samples['answer'])
        print('pred', pred)
        return {'pred': pred}
    def inference1(self, embeds,question):

        question = self.tokenizer(question, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(
            self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        batch_inputs_embeds = []
        batch_attention_mask = []
        input_ids = question.input_ids + eos_user_tokens.input_ids
        inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
        inputs_embeds = torch.cat([bos_embeds, embeds, inputs_embeds], dim=0)
        batch_inputs_embeds.append(inputs_embeds)
        batch_attention_mask.append([1] * inputs_embeds.shape[0])
        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                # do_sample=True,
                use_cache=True,  

            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return pred

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param
