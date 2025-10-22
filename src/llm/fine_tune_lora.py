import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
import torch

MODEL_NAME = os.environ.get('BASE_MODEL', 'tiiuae/falcon-7b')
OUTPUT_DIR = 'models/lora-chatlegal'

def train_lora(dataset_path: str, epochs=3, batch_size=8):
    ds = load_dataset('json', data_files=dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, load_in_8bit=True, device_map='auto')

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=['q_proj','v_proj'],
        lora_dropout=0.05,
        bias='none',
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, peft_config)

    def tokenize_fn(examples):
        return tokenizer(examples['text'], truncation=True, padding='longest')

    ds = ds.map(tokenize_fn, batched=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        logging_steps=10,
        save_total_limit=2,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'] if 'train' in ds else ds['train']
    )
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--epochs', type=int, default=1)
    args = parser.parse_args()
    train_lora(args.dataset, epochs=args.epochs)
