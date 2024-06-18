import torch
from loguru import logger
from constants import DEVICE, OUTPUT_DIR
from sklearn.metrics import accuracy_score
import numpy as np
import json
from models import MisMISDataset
from torch.utils.data import DataLoader
from constants import BATCH_SIZE
import pandas as pd
import os
from pathlib import Path

AGN_BEST_ACC = 0.0
AWR_BEST_ACC = 0.0

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(Path(dir), exist_ok=True)

def format_input_template(source, target):
    return f"source: {source}\ntarget: {target}"

def load_shroom_val(path):
    with open(path, 'r') as f:
        shroom_val = json.load(f)
    
    hyp = [sample['hyp'] for sample in shroom_val]
    tgt = [sample['tgt'] if sample['task'] != 'PG' else sample['src'] for sample in shroom_val]
    labels = [sample['label'] for sample in shroom_val]
    
    dataset = MisMISDataset(hyp, tgt, [0 if label == 'Hallucination' else 1 for label in labels])
    return DataLoader(dataset, batch_size=BATCH_SIZE)

def save_model_checkpoint(model, checkpoint_name):
    peft_path = OUTPUT_DIR / checkpoint_name / 'peft_encoder'
    create_dir(peft_path)

    classification_head_path = OUTPUT_DIR / checkpoint_name / 'classification_head'
    create_dir(classification_head_path)

    model.encoder.save_pretrained(peft_path)
    torch.save(model.classification_head.state_dict(), classification_head_path / 'classification_model.pt')


def evaluate(model, tokenizer, direct, loss, type):
    global AGN_BEST_ACC
    global AWR_BEST_ACC
    model.eval()

    preds, trues, losses = [], [], []
    sources_lst = []
    targets_lst = []


    for batch in direct:
        sources, targets, true_labels = batch
        sources_lst.extend(sources)
        targets_lst.extend(targets)

        with torch.no_grad():
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

            proba = torch.sigmoid(proba)

        proba = torch.round(proba)

        loss_value = loss(proba, true_labels.to(torch.float32).to(DEVICE).unsqueeze(1)).cpu().numpy()
        losses.append(loss_value)

        preds.extend(proba.cpu().numpy())
        trues.extend(true_labels)

    trues = [el.item() for el in trues]
    preds = [int(el[0] > 0.5) for el in preds]

    accuracy = accuracy_score(trues, preds)

    meta_df = pd.DataFrame({
        'source': sources_lst,
        'target': targets_lst,
        'true_label': trues,
        'preds': preds
    })
    if type =='awr':
        if accuracy > AWR_BEST_ACC:
            logger.info(f"New best AWR ACC={accuracy}")
            meta_df.to_csv(f'/app/res/awr_{str(round(accuracy, 2))}')
            AWR_BEST_ACC = accuracy

            save_model_checkpoint(model, f'awr_acc_{str(round(accuracy, 2))}')
    elif type =='agn':
        if accuracy > AGN_BEST_ACC:
            logger.info(f"New best AGN ACC={accuracy}")
            meta_df.to_csv(f'/app/res/agn_{str(round(accuracy, 2))}')
            AGN_BEST_ACC = accuracy

            save_model_checkpoint(model, f'agn_acc_{str(round(accuracy, 2))}')
    return accuracy, np.mean(losses)
