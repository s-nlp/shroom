from torch.utils.data import Dataset
from torch import nn
import torch
from torch import Tensor

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

class MisMISDataset(Dataset):    
    def __init__(self, queries, answers, labels):
        self.queries = queries
        self.answers = answers
        self.labels = labels
        
        self.task = 'Retrieve semantically similar text.'
        
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        return get_detailed_instruct(self.task, self.queries[idx]), self.answers[idx], self.labels[idx]
        # return self.queries[idx], self.answers[idx], self.labels[idx]

class ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.pre_head = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.head = nn.Linear(config.hidden_size // 2, 1)
    
    def forward(self, inputs):
        inputs = self.dropout(inputs)
        inputs = self.dense(inputs)
        inputs = torch.tanh(inputs)
        inputs = self.dropout(inputs)
        inputs = self.pre_head(inputs)
        return self.head(inputs)


class MistralCrossEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.classification_head = ClassificationHead(encoder.config)

    def forward(self, input_batch):
        outputs = self.encoder(**input_batch)
        last_token_pooled_emb = self.last_token_pool(outputs.last_hidden_state, input_batch['attention_mask'])
        return self.classification_head(last_token_pooled_emb)

    @staticmethod
    def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

# class MistralCrossEncoder(nn.Module):
#     def __init__(self, encoder):
#         super().__init__()
#         self.encoder = encoder
#         self.dense = nn.Linear(encoder.config.hidden_size, encoder.config.hidden_size)
#         self.dropout = nn.Dropout(0.1)
#         self.head = nn.Linear(encoder.config.hidden_size * 2, 1)

#     def forward(self, first_input_batch, second_input_batch):
#         first_outputs = self.encoder(**first_input_batch)
#         first_outputs = self.dropout(self.last_token_pool(first_outputs.last_hidden_state, first_input_batch['attention_mask']))
#         first_outputs = self.dense(first_outputs)

#         second_outputs = self.encoder(**second_input_batch)
#         second_outputs = self.dropout(self.last_token_pool(second_outputs.last_hidden_state, second_input_batch['attention_mask']))
#         second_outputs = self.dense(second_outputs)

#         outputs = torch.cat([first_outputs, second_outputs], -1)
#         return self.head(outputs)

#     @staticmethod
#     def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
#         left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
#         if left_padding:
#             return last_hidden_states[:, -1]
#         else:
#             sequence_lengths = attention_mask.sum(dim=1) - 1
#             batch_size = last_hidden_states.shape[0]
#             return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
