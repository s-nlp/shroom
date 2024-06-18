import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import datasets
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from models import MisMISDataset, MistralCrossEncoder
from constants import BATCH_SIZE, MODEL_NAME, DEVICE, NUM_EPOCHS, EVAL_STEPS, GRAD_ACCUM_STEPS, LEARNING_RATE
from func import evaluate, load_shroom_val, format_input_template
from peft import LoraConfig, get_peft_model
from loguru import logger
import wandb
import os
from pathlib import Path

outdir = '/app/res'
if not os.path.exists(outdir):
    os.makedirs(Path(outdir), exist_ok=True)

wandb.init(config={})

encoder_model = AutoModel.from_pretrained(MODEL_NAME)
encoder_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

peft_config = LoraConfig(inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1)
peft_model = get_peft_model(encoder_model, peft_config)

model = MistralCrossEncoder(encoder=peft_model)
model.to(DEVICE);

wandb.watch(model, log_freq=100)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# QQP
quora = datasets.load_dataset('quora') 

source_text = [el['text'][0] for el in quora['train']['questions']]
gener_text = [el['text'][1] for el in quora['train']['questions']]
label = np.array(quora['train']['is_duplicate']).astype(int)

qqp_df = pd.DataFrame({'source': source_text, 
                       "target": gener_text, 
                       "label": label, 
                       "from": "qqp"})

qqp_df = qqp_df.sample(10_000, random_state=42)

# SYNT LLaMA
llama_synt = pd.read_csv('/app/data/llama_synt.csv')[['source', 'target', 'label']]
llama_synt['from'] = 'llama'
aware_llama_synt = pd.read_csv('/app/data/aware_llama_synt.csv')[['source', 'target', 'label', 'from']]

# SYNT GPT v3
# gpt_synt_incorrect = pd.read_csv('/app/data/incorrect_gpt_synt.csv')
# gpt_synt_correct = pd.read_csv('/app/data/correct_gpt_synt.csv')

# df_all = pd.concat([qqp_df, llama_synt, gpt_synt_incorrect, gpt_synt_correct])
# df_all = pd.concat([qqp_df, llama_synt])
df_all = pd.concat([qqp_df, aware_llama_synt, llama_synt])
df_all = df_all.sample(frac=1, random_state=42)

train_indx = int(len(df_all) * 0.9)
val_indx = int(len(df_all) * 0.95)

df_train = df_all[:train_indx]
df_val = df_all[train_indx:val_indx]
df_test = df_all[val_indx:]


loaders_dict = {}

for df, dftype in zip([df_train,df_val, df_test], ['train','val','test']):
    dataset = MisMISDataset(df['source'].tolist(), df["target"].tolist(), df["label"].tolist())
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    loaders_dict[dftype] = {'direct': dataloader}
    
    if dftype == 'val':
        split_index = 500
        df = df[:split_index]

        dataset = MisMISDataset(df['source'].tolist(), df["target"].tolist(), df["label"].tolist())
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

        loaders_dict[f'{dftype}_toy'] = {'direct':dataloader}


aware_shroom_val = load_shroom_val("/app/data/val/val.model-aware.v2.json")
agn_shroom_val = load_shroom_val("/app/data/val/val.model-agnostic.json")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCEWithLogitsLoss()

for epoch in range(NUM_EPOCHS):
    wandb.log({'train/epoch': epoch + 1})
    logger.info(f'Epoch: {epoch}')

    tq = tqdm(loaders_dict['train']['direct'], total=len(loaders_dict['train']['direct']))

    loss_list = []

    awr_shroom_acc, awr_shroom_mean_loss = evaluate(model, tokenizer, aware_shroom_val, criterion, type='awr')
    agn_shroom_acc, agn_shroom_mean_loss = evaluate(model, tokenizer, agn_shroom_val, criterion, type='agn')
    mixin_acc, mixin_mean_loss = evaluate(model, tokenizer, loaders_dict['val_toy']['direct'], criterion, type='mixin')

    logger.info(f'epoch={epoch} step=0 mix_acc={round(mixin_acc, 2)} mix_loss={round(mixin_mean_loss, 2)} awr_acc={round(awr_shroom_acc, 2)} awr_loss={round(awr_shroom_mean_loss, 2)}')
    wandb.log({'val/mix_acc': mixin_acc, 'val/mix_loss': mixin_mean_loss, 'val/awr_acc': awr_shroom_acc, 'val/awr_loss': awr_shroom_mean_loss, 'val/agn_acc': agn_shroom_acc, 'val/agn_loss': agn_shroom_mean_loss})

    for i, batch in enumerate(tq):
        model.train()

        sources, targets, labels = batch

        inputs = [format_input_template(source, target) for source, target in zip(sources, targets)]
        # first_inputs = [format_input_template(source, target) for source, target in zip(sources, targets)]
        # second_inputs = [format_input_template(target, source) for source, target in zip(sources, targets)]
        
        tokenized = tokenizer(inputs, return_attention_mask=False, padding=True, truncation=False)
        tokenized['input_ids'] = [input_ids + [tokenizer.eos_token_id] for input_ids in tokenized['input_ids']]
        tokenized = tokenizer.pad(tokenized, padding=True, return_attention_mask=True, return_tensors='pt').to(DEVICE)

        # first_tokenized = tokenizer(first_inputs, return_attention_mask=False, padding=True, truncation=False)
        # first_tokenized['input_ids'] = [input_ids + [tokenizer.eos_token_id] for input_ids in first_tokenized['input_ids']]
        # first_tokenized = tokenizer.pad(first_tokenized, padding=True, return_attention_mask=True, return_tensors='pt').to(DEVICE)
        
        # second_tokenized = tokenizer(second_inputs, return_attention_mask=False, padding=True, truncation=False)
        # second_tokenized['input_ids'] = [input_ids + [tokenizer.eos_token_id] for input_ids in second_tokenized['input_ids']]
        # second_tokenized = tokenizer.pad(second_tokenized, padding=True, return_attention_mask=True, return_tensors='pt').to(DEVICE)

        proba = model(tokenized)
        # proba = model(first_tokenized, second_tokenized)

        true_labels = labels.to(torch.float32).to(DEVICE).unsqueeze(1)

        output = criterion(proba, true_labels)      
        loss_list.append(output.item())
        output.backward()

        if i % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

        last_loss = np.mean(loss_list)
        tq.set_description(f'loss: {last_loss:4.4f}')

        wandb.log({'train/loss': last_loss})

        if i % EVAL_STEPS == 0 and i != 0:
            awr_shroom_acc, awr_shroom_mean_loss = evaluate(model, tokenizer, aware_shroom_val, criterion, type='awr')
            agn_shroom_acc, agn_shroom_mean_loss = evaluate(model, tokenizer, agn_shroom_val, criterion,  type='agn')
            mixin_acc, mixin_mean_loss = evaluate(model, tokenizer, loaders_dict['val_toy']['direct'], criterion, type='mixin')

            logger.info(f'epoch={epoch:} step={i} mix_acc={round(mixin_acc, 2)} mix_loss={round(mixin_mean_loss, 2)} awr_acc={round(awr_shroom_acc, 2)} awr_loss={round(awr_shroom_mean_loss, 2)}')
            wandb.log({'val/mix_acc': mixin_acc, 'val/mix_loss': mixin_mean_loss, 'val/awr_acc': awr_shroom_acc, 'val/awr_loss': awr_shroom_mean_loss, 'val/agn_acc': agn_shroom_acc, 'val/agn_loss': agn_shroom_mean_loss})
