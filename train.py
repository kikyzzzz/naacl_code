import os
import argparse
from datasets import load_dataset, load_from_disk
from trl import SFTTrainer
from trl.trainer import SFTConfig
from peft import LoraConfig
from transformers import AutoTokenizer
from collate_fn import CustomizedDataCollatorForChatML
import torch
from accelerate import PartialState
from transformers import AutoTokenizer, AutoModelForCausalLM

import numpy as np

from accelerate import Accelerator
original_unwrap_model = Accelerator.unwrap_model


def fixed_unwrap_model(self, model, **kwargs):
    kwargs.pop("keep_torch_compile", None)
    return original_unwrap_model(self, model, **kwargs)

Accelerator.unwrap_model = fixed_unwrap_model

parser = argparse.ArgumentParser(description='Train model with custom learning rate and threshold')
parser.add_argument('--learning_rate', type=float, default=2e-5,
                    help='Learning rate for training (default: 2e-5)')
parser.add_argument('--threshold', type=float, default=0,
                    help='Threshold for data collator (default: 2.5)')
parser.add_argument('--dataset', type=str, default="math",
                    help='Data task to train (default: mbpp, gsm8k,math,arc)')
parser.add_argument('--target_model', type=str, default="", help='base model path to train')
parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
parser.add_argument("--data_mode", type=str, help="data", default="distill")
parser.add_argument("--ppl_threshold", type=int, help="(default: simple, base, hard, ppl)", default=12)
parser.add_argument("--threshold_dist", type=int, help="(default: simple, base, hard, ppl)", default=0.2)
parser.add_argument("--threshold_skew", type=int, help="(default: simple, base, hard, ppl)", default=12)
parser.add_argument("--answer_length", type=int, help="(default: simple, base, hard, ppl)", default=250)
parser.add_argument("--train_mode", type=str, help="(default: simple, base, hard, origin)", default="base")
parser.add_argument("--seed", type=int, default=42, help="Random seed for training (set in SFTConfig and path)")

args_cli = parser.parse_args()


max_seq_length = 512
if 'Llama-3-8B' in args_cli.target_model:
    model_name = "llama-8b"
elif 'Llama' in args_cli.target_model:
    model_name = "llama-3b"
elif 'Qwen' in args_cli.target_model:
    model_name = "qwen-4b"
elif 'gemma' in args_cli.target_model:
    model_name = "gemma-2b"
target_model = args_cli.target_model
dataset_name = args_cli.dataset
train_mode = args_cli.train_mode
threshold = str(args_cli.threshold).replace(".", "-")
output_dir = f"{model_name}/{dataset_name}_config_lr{args_cli.learning_rate}_t{threshold}"
data_path = os.path.join('dataset/stm/', target_model.split('/')[-2] + '_' + dataset_name)
data_path = data_path + '_' + args_cli.data_mode

output_dir = output_dir + '_' + args_cli.data_mode + '_s' + str(args_cli.seed)
if train_mode == "first_token_mask":
    output_dir += "_first_token_mask"
dataset = load_from_disk(data_path)
example_dataset = dataset[1]
example_data = dataset[0]

device_string = PartialState().process_index
tokenizer = AutoTokenizer.from_pretrained(target_model)
tokenizer.pad_token = tokenizer.eos_token
target_model = AutoModelForCausalLM.from_pretrained(
    args_cli.target_model,
    torch_dtype=torch.bfloat16,
    device_map='cuda'
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    use_dora=True,
)
threshold = str(args_cli.threshold).replace(".", "-")
args = SFTConfig(output_dir=output_dir,
    gradient_checkpointing=True,
    num_train_epochs=3,
    learning_rate=args_cli.learning_rate,
    save_strategy="epoch",
    save_total_limit=None,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    bf16=True,
    warmup_ratio=0.05,
    logging_steps=100,
    packing=False,
    lr_scheduler_type="cosine",
    local_rank=args_cli.local_rank,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim='adamw_bnb_8bit',
    dataset_kwargs={"skip_prepare_dataset": True},
    seed=args_cli.seed,
)
args.remove_unused_columns = False
trainer = SFTTrainer(
    target_model,
    train_dataset=dataset,
    args=args,
    peft_config=peft_config,
    data_collator=CustomizedDataCollatorForChatML(tokenizer, threshold=args_cli.threshold, apply="highest", train_mode=train_mode)
)
trainer.train()