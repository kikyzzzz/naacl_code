import os
import re
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

DEFAULT_MODEL_PATH = ""
DEFAULT_LOAD_ADAPTER = None
DEFAULT_MATH_DATA_PATH = ""
DEFAULT_OUTPUT_DIR = ""
DEFAULT_MATH_EXISTING_PATH = None

MAX_LENGTH = 1024
GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.0,
    "do_sample": False,
    "eos_token_id": None,
    "pad_token_id": None,
}
HYBRID_FIRST_TOKENS = 100

MESSAGES_KEY = "conversation"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate self-answer for Math dataset using LLMs (with re-generation for wrong answers)")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to base model")
    parser.add_argument("--load_adapter", type=str, default=DEFAULT_LOAD_ADAPTER,
                        help="Path to LoRA adapter (optional)")
    parser.add_argument("--data_path", type=str, default="", help="Path to dataset (auto-set if empty)")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save results")
    parser.add_argument("--existing_dataset_path", type=str,
                        default=DEFAULT_MATH_EXISTING_PATH,
                        help="Path to existing generated dataset")
    parser.add_argument("--max_retry", type=int, default=0, help="Max retry times for re-generation")
    parser.add_argument("--stm", type=str, default="None", help="STM flag (for output filename)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--math_sample_num", type=int, default=2000,
                        help="Number of Math dataset samples to process (default: 100000)")
    parser.add_argument("--distill_type", type=str, default='distill', help="base、distill、hybrid、repeat_answer")
    return parser.parse_args()


def clean_latex_symbols(s):
    if s is None or s == "":
        return ""
    s = re.sub(r'\$', '', s)
    s = re.sub(r'\\left', '', s)
    s = re.sub(r'\\right', '', s)
    s = s.replace(" ", "").strip()
    return s


def extract_math_numeric_answer(answer_str):
    try:
        pattern = r'\\boxed\{(?:[^{}]|\{[^{}]*\})*\}'
        matches = re.findall(pattern, answer_str)
        if not matches:
            return ""
        boxed_content = matches[-1].replace("\\boxed{", "")
        if boxed_content.endswith('}'):
            boxed_content = boxed_content[:-1]
        return boxed_content
    except Exception:
        return ""


def build_math_conversation(example):
    instruction = """
You are given a grade school math question. Please answer the question in the following format:

Q: <Question>
A: <Think step by step here> \\boxed{{<number only answer>}}

Format requirements : you must first output your reasoning before finalized with the "\\boxed{{}}" format followed by the final numeric answer

Reference Answer:
{reference_answer}

Now answer the question yourself:
"""
    instruction = instruction.format(question=example['problem'], reference_answer=example['solution'])
    human_content = f"{instruction}\nQ: {example['problem'].strip()}\nA: "

    conversation = [
        {"from": "human", "value": human_content},
    ]
    example[MESSAGES_KEY] = conversation
    return example


def build_math_conversation_base(example):
    instruction = """
You are given a grade school math question. Please answer the question in the following format:

Q: <Question>
A: <Think step by step here> \\boxed{{<number only answer>}}

Format requirements : you must first output your reasoning before finalized with the "\\boxed{{}}" format followed by the final numeric answer
        """
    human_content = f"{instruction}\nQ: {example['problem'].strip()}\nA: "

    conversation = [
        {"from": "human", "value": human_content},
    ]
    example[MESSAGES_KEY] = conversation
    return example


def build_math_conversation_repeat(example):
    instruction = """You are given a grade school math question and its correct reference answer. Please answer the question in the following format:

<Think step by step here> \\boxed{{<number only answer>}}

Format requirements : you must first output your reasoning before finalized with the "\\boxed{{}}" format followed by the final numeric answer

Reference Answer:{reference_answer}
                """
    instruction = instruction.format(question=example['problem'], reference_answer=example['solution'])
    reference_answer = example['solution'],
    human_content = f"{instruction}\nQ: {example['problem'].strip()}\nA: "

    conversation = [
        {"from": "human", "value": human_content},
    ]
    example[MESSAGES_KEY] = conversation
    return example


def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side="right",
        use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        device_map="auto" if args.device == "cuda" else "cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    if args.load_adapter and os.path.exists(args.load_adapter):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.load_adapter)

    model.eval()
    return model, tokenizer


def generate_self_answer(example, model, tokenizer, args, retry_count=0):
    try:
        if args.distill_type == "hybrid":
            distill_full_answer = example.get("self_answer", "")
            if not distill_full_answer:
                example["self_answer"] = ""
                example["self_numeric_answer"] = ""
                example["is_correct"] = False
                return example

            distill_tokens = tokenizer(
                distill_full_answer,
                truncation=True,
                max_length=HYBRID_FIRST_TOKENS,
                padding=False,
                return_tensors="pt"
            ).input_ids.to(args.device)

            partial_output = tokenizer.decode(
                distill_tokens[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            ).strip()

            base_conv = build_math_conversation_base(example.copy())[MESSAGES_KEY][0]
            user_prompt = base_conv['value']

            formatted_prompt = [{'role': 'user', 'content': user_prompt}]
            chat_template = tokenizer.apply_chat_template(
                formatted_prompt,
                tokenize=False,
                add_generation_prompt=True,
                system_message=""
            )

            chat_template = chat_template + partial_output

            tokenized_prompt = tokenizer(
                chat_template,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=False,
                return_tensors='pt',
                add_special_tokens=False
            ).to(args.device)

            remain_tokens = GENERATION_CONFIG["max_new_tokens"] - HYBRID_FIRST_TOKENS
            with torch.no_grad():
                generation_output = model.generate(
                    **tokenized_prompt,
                    max_new_tokens=remain_tokens,
                    temperature=GENERATION_CONFIG["temperature"],
                    do_sample=GENERATION_CONFIG["do_sample"],
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

            prompt_len = tokenized_prompt.input_ids.shape[1]
            generated_tokens = generation_output[0][prompt_len:]
            new_suffix = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            ).strip()

            self_output = f"{partial_output} {new_suffix}".strip()

        elif args.distill_type == "repeat_answer":
            prompt = example[MESSAGES_KEY][0]
            formatted_prompt = [{'role': 'user', 'content': prompt['value']}]
            chat_template = tokenizer.apply_chat_template(
                formatted_prompt, tokenize=False, add_generation_prompt=True, system_message=""
            )
            fixed_prefix = f"Q: {example['problem']}\n\nA: "
            chat_template = chat_template + fixed_prefix
            tokenized_prompt = tokenizer(
                chat_template, truncation=True, max_length=MAX_LENGTH, padding=False, return_tensors='pt'
            ).to(args.device)

            with torch.no_grad():
                generation_output = model.generate(
                    **tokenized_prompt,
                    max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
                    temperature=GENERATION_CONFIG["temperature"],
                    do_sample=GENERATION_CONFIG["do_sample"],
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_len = tokenized_prompt.input_ids.shape[1]
            generated_tokens = generation_output[0][prompt_len:]
            self_output = tokenizer.decode(
                generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
            ).strip()
            self_output = fixed_prefix + self_output
        else:
            prompt = example[MESSAGES_KEY][0]
            formatted_prompt = [{'role': 'user', 'content': prompt['value']}]
            if "Qwen" not in args.model_path:
                chat_template = tokenizer.apply_chat_template(
                    formatted_prompt, tokenize=False, add_generation_prompt=True, system_message=""
                )
            else:
                chat_template = tokenizer.apply_chat_template(
                    formatted_prompt, tokenize=False, add_generation_prompt=True, system_message="", enable_thinking=False
                )
            tokenized_prompt = tokenizer(
                chat_template, truncation=True, max_length=MAX_LENGTH, padding=False, return_tensors='pt', add_special_tokens=False
            ).to(args.device)

            with torch.no_grad():
                generation_output = model.generate(
                    **tokenized_prompt,
                    max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
                    temperature=GENERATION_CONFIG["temperature"],
                    do_sample=GENERATION_CONFIG["do_sample"],
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_len = tokenized_prompt.input_ids.shape[1]
            generated_tokens = generation_output[0][prompt_len:]
            self_output = tokenizer.decode(
                generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
            ).strip()

        self_numeric = extract_math_numeric_answer(self_output)
        gold_numeric = example["answer"]

        if self_numeric != gold_numeric and retry_count < args.max_retry:
            return generate_self_answer(example, model, tokenizer, args, retry_count + 1)

        if args.distill_type != "hybrid":
            example["self_answer"] = self_output
            example["self_numeric_answer"] = self_numeric
            example["is_correct"] = (self_numeric == gold_numeric)
        else:
            example["self_hybrid_answer"] = self_output
            example["self_numeric_hybrid_answer"] = self_numeric
            example["is_hybrid_correct"] = (self_numeric == gold_numeric)

    except Exception as e:
        example["self_answer"] = ""
        example["self_numeric_answer"] = ""
        example["is_correct"] = False

    return example


def validate_and_regenerate(dataset, model, tokenizer, args):
    dataset = dataset.map(
        build_math_conversation,
        num_proc=4,
        desc="Rebuilding conversation field"
    )

    processed_dataset = []
    wrong_count = 0
    regenerated_count = 0

    for example in tqdm(dataset, desc="Validating and re-generating answers", dynamic_ncols=True):
        self_numeric = extract_math_numeric_answer(example.get("self_answer", ""))
        gold_numeric = example["answer"]

        if self_numeric != gold_numeric:
            wrong_count += 1
            example = generate_self_answer(example, model, tokenizer, args)
            if example["is_correct"]:
                regenerated_count += 1

        processed_dataset.append(example)

    return Dataset.from_list(processed_dataset)


def filter_by_answer_consistency(dataset, args):
    consistent_samples = []
    inconsistent_samples = []

    for example in tqdm(dataset, desc="Filtering samples", dynamic_ncols=True):
        if args.distill_type != "hybrid":
            self_numeric = example['self_numeric_answer']
        else:
            self_numeric = example['self_numeric_hybrid_answer']
        gold_numeric = example["answer"]

        if self_numeric and gold_numeric and self_numeric == gold_numeric:
            consistent_samples.append(example)
        else:
            inconsistent_samples.append(example)

    consistent_dataset = Dataset.from_list(consistent_samples)
    inconsistent_dataset = Dataset.from_list(inconsistent_samples)

    return consistent_dataset, inconsistent_dataset


def main():
    args = parse_args()

    if not args.data_path:
        args.data_path = DEFAULT_MATH_DATA_PATH
    if not args.existing_dataset_path:
        args.existing_dataset_path = DEFAULT_MATH_EXISTING_PATH

    model, tokenizer = load_model_and_tokenizer(args)

    if args.existing_dataset_path and os.path.exists(args.existing_dataset_path) and args.distill_type != 'hybrid':
        dataset = Dataset.load_from_disk(args.existing_dataset_path)
    elif args.distill_type == 'hybrid':
        dataset = Dataset.load_from_disk(args.existing_dataset_path)

        if isinstance(dataset, Dataset):
            dataset = dataset.select(range(min(args.math_sample_num, len(dataset))))
        else:
            dataset = Dataset.from_dict(dataset)
            dataset = dataset.select(range(min(args.math_sample_num, len(dataset))))

        processed_dataset = []
        for example in tqdm(dataset, desc="Generating self-answers", dynamic_ncols=True):
            processed_example = generate_self_answer(example, model, tokenizer, args)
            processed_dataset.append(processed_example)
        dataset = Dataset.from_list(processed_dataset)
    else:
        full_dataset = load_dataset(args.data_path)
        dataset = full_dataset["train"]

        if isinstance(dataset, Dataset):
            dataset = dataset.select(range(min(args.math_sample_num, len(dataset))))
        else:
            dataset = Dataset.from_dict(dataset)
            dataset = dataset.select(range(min(args.math_sample_num, len(dataset))))

        if args.distill_type == "base":
            dataset = dataset.map(build_math_conversation_base, num_proc=4, desc="Building base conversation")
        elif args.distill_type == "distill":
            dataset = dataset.map(build_math_conversation, num_proc=4, desc="Building distill conversation")
        elif args.distill_type == "repeat_answer":
            dataset = dataset.map(build_math_conversation_repeat, num_proc=4, desc="Building distill conversation")
        elif args.distill_type == "hybrid":
            pass

        processed_dataset = []
        for example in tqdm(dataset, desc="Generating self-answers", dynamic_ncols=True):
            processed_example = generate_self_answer(example, model, tokenizer, args)
            processed_dataset.append(processed_example)
        dataset = Dataset.from_list(processed_dataset)

    consistent_dataset, inconsistent_dataset = filter_by_answer_consistency(dataset, args)

    output_file = args.model_path.split("/")[-2] + '_math_self_answer' + f"_{args.distill_type}"
    if args.distill_type == "hybrid":
        output_file += f'_{HYBRID_FIRST_TOKENS}'
    if args.load_adapter:
        lr = args.load_adapter.split("/")[-3].split("_t")[-1]
        step = args.load_adapter.split("/")[-2].split("-")[-1]
        output_file += '_' + lr + '_' + step
    if args.stm != "None":
        output_file += '_' + args.stm

    output_file += f"_sample{args.math_sample_num}"
    save_dir = os.path.join(args.output_dir, output_file)
    filter_dir = os.path.join(args.output_dir, output_file + '_filter')
    infilter_dir = os.path.join(args.output_dir, output_file + '_infilter')
    os.makedirs(save_dir, exist_ok=True)
    consistent_dataset.save_to_disk(filter_dir)
    inconsistent_dataset.save_to_disk(infilter_dir)

    total_samples = len(dataset)
    total_success = sum(1 for ex in dataset if ex["self_answer"] != "")
    total_correct = sum(1 for ex in dataset if ex.get("is_correct", False))
    accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0


if __name__ == "__main__":
    main()