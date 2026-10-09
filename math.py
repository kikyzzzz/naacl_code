import os
import re
import json
import datetime
import argparse

import httpx
import torch
import numpy as np
from openai import OpenAI
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel

SANKUAI_APPID = ""
SANKUAI_BASE_URL = ""
client = OpenAI(
    api_key=SANKUAI_APPID,
    base_url=SANKUAI_BASE_URL,
)
client_qwen = OpenAI(
    api_key=SANKUAI_APPID,
    base_url=SANKUAI_BASE_URL,
)


def is_equiv_by_api(pred_answer, gold_answer):
    if pred_answer is None or gold_answer is None:
        return False

    system_prompt = """
You are a strict mathematical evaluation expert. Your task is to determine whether two mathematical answers are semantically equivalent.
Rules:
1. Ignore LaTeX formatting differences (e.g., \\frac{\\pi}{2} vs π/2, \\left/\\right, $ symbols, spaces, etc.);
2. Numerical equivalence is sufficient (e.g., 2 and 2.0, 1/2 and 0.5, \\pi/2 and 1.5707963);
3. Coordinate pairs/expressions must be equivalent part by part (e.g., (3,\\pi/2) and (3,1.5707963) are equivalent);
4. Return only "True" or "False", without any additional content.
    """.strip()

    user_prompt = f"""
Please determine whether the following two mathematical answers are equivalent:
Model generated answer: {pred_answer}
Standard answer: {gold_answer}
Return only "True" or "False", without any other content.
    """.strip()

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
            temperature=0.0
        )
        result = response.choices[0].message.content.strip()
        return result.lower() == "true"
    except Exception as e:
        return False


def clean_latex_symbols(s):
    if s is None:
        return None
    s = re.sub(r'\$', '', s)
    s = re.sub(r'\\left', '', s)
    s = re.sub(r'\\right', '', s)
    s = s.replace(" ", "").strip()
    return s


def remove_boxed(s):
    if s is None:
        return None
    left = "\\boxed{"
    boxed_match = re.search(r'\\boxed\{', s)
    if not boxed_match:
        return None
    start_idx = boxed_match.end()
    end_idx = None
    brace_count = 0
    for i in range(start_idx, len(s)):
        if s[i] == '{':
            brace_count += 1
        elif s[i] == '}':
            if brace_count == 0:
                end_idx = i
                break
            brace_count -= 1
    if end_idx is None:
        return None
    content = s[start_idx:end_idx]
    return clean_latex_symbols(content)


def last_boxed_only_string(s):
    if s is None or s == "":
        return None
    pattern = r'\\boxed\{(?:[^{}]|\{[^{}]*\})*\}'
    matches = re.findall(pattern, s)
    if not matches:
        return None
    last_match = matches[-1]
    return remove_boxed(last_match)


def clean_numbers(s):
    if s is None or s == "":
        return None
    s = s.replace('\\\\', '\\')
    s = s.replace("$", "").replace(",", "，").strip()
    s = re.sub(r'\s+', '', s)
    return s


def normalize_latex_expr(s):
    if s is None:
        return s

    s = s.replace('\\\\frac', '\\frac')
    s = s.replace('\\\\pi', '\\pi')
    s = s.replace('\\\\left', '\\left')
    s = s.replace('\\\\right', '\\right')

    s = re.sub(r'(\d+)\\frac\{\\pi\}\{(\d+)\}', r'\1*π/\2', s)
    s = re.sub(r'\\frac\{\\pi\}\{(\d+)\}', r'π/\1', s)
    s = re.sub(r'(\d+)\\frac\{(\d+)\}\{(\d+)\}', r'\1*\2/\3', s)
    s = re.sub(r'\\frac\{(\d+)\}\{(\d+)\}', r'\1/\2', s)

    s = s.replace('，', ',')
    s = s.replace('\\left', '').replace('\\right', '')
    s = s.replace('(', '').replace(')', '')
    s = s.replace('π', 'π').strip()

    return s


def is_equiv(pred, gold, tol=1e-6):
    if pred is None or gold is None:
        return False

    pred_clean = clean_numbers(pred)
    gold_clean = clean_numbers(gold)

    pred_norm = normalize_latex_expr(pred_clean)
    gold_norm = normalize_latex_expr(gold_clean)

    try:
        def safe_eval(expr):
            expr = expr.replace('π', '3.1415926')
            allowed = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/', '.', ','}
            if not all(c in allowed for c in expr if c != ' '):
                raise ValueError("non-numeric expression")
            if ',' in expr:
                parts = expr.split(',')
                return [eval(p) for p in parts]
            return eval(expr)

        pred_val = safe_eval(pred_norm)
        gold_val = safe_eval(gold_norm)

        if isinstance(pred_val, list) and isinstance(gold_val, list):
            return all(abs(p - g) < tol for p, g in zip(pred_val, gold_val))
        else:
            return abs(pred_val - gold_val) < tol
    except:
        return pred_norm == gold_norm


def get_level_type(row):
    level = row.get('level', None)
    prob_type = row.get('subject', None)
    try:
        if level:
            level = int(level)
    except:
        level = None
    return level, prob_type


def create_unique_filename(base_name):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_eval_{timestamp}.jsonl"


QUERY_TEMPLATE = """You are given a grade school math question. Please answer the question in the following format:

        Q: <Question>
        A: <Think step by step here> \\boxed{{<number only answer>}}

        Format requirements : you must first output your reasoning before finalized with the "\\boxed{{}}" format followed by the final numeric answer
"""


def main(args):
    log_dir = Path(args.result_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.result_path is not None and Path(args.result_path).exists():
        import json

        total = 0
        correct_cnt = 0
        skipped = 0
        fmt_fail_cnt = 0

        cors = {}
        level_cors = {}
        subject_cors = {}
        subjects = ['Prealgebra', 'Algebra', 'Number Theory', 'Counting & Probability', 'Geometry',
                    'Intermediate Algebra',
                    'Precalculus']

        model_name = Path(args.result_path).stem
        score_path = log_dir / f"{model_name}_scores.txt"

        with open(args.result_path, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                try:
                    entry = json.loads(line.strip())
                except:
                    continue

                total += 1

                pred_answer = entry.get("pred_answer", None)
                gold_answer = entry.get("gold_answer", None)
                correct = is_equiv_by_api(pred_answer, gold_answer)

                if correct:
                    correct_cnt += 1
                if pred_answer is None:
                    skipped += 1
                    fmt_fail_cnt += 1

                prob_level = entry.get("level")
                prob_type = entry.get("type")

                if (prob_level, prob_type) in cors:
                    cors[(prob_level, prob_type)].append(correct)
                else:
                    cors[(prob_level, prob_type)] = [correct]

                if prob_level is not None:
                    if prob_level in level_cors:
                        level_cors[prob_level].append(correct)
                    else:
                        level_cors[prob_level] = [correct]

                if prob_type is not None:
                    if prob_type in subject_cors:
                        subject_cors[prob_type].append(correct)
                    else:
                        subject_cors[prob_type] = [correct]

        final_accuracy = correct_cnt / total if total > 0 else 0
        fmt_fail_rate = fmt_fail_cnt / total if total > 0 else 0
        skip_rate = skipped / (total + skipped) if (total + skipped) > 0 else 0

        with open(score_path, "w", encoding="utf-8") as f:
            f.write("===== Subject + Difficulty Accuracy =====\n")
            for prob_type in subjects:
                for prob_level in [1, 2, 3, 4, 5]:
                    if (prob_level, prob_type) in cors:
                        cors_list = cors[(prob_level, prob_type)]
                        acc = np.mean(cors_list)
                        cnt = np.sum(cors_list)
                        total_cnt = len(cors_list)
                        log_str = f"{prob_type} Level {prob_level} Accuracy = {cnt}/{total_cnt} = {acc:.3f}"
                        f.write(log_str + "\n")

            f.write("\n===== Difficulty Accuracy =====\n")
            for level in sorted(level_cors):
                cors_list = level_cors[level]
                acc = np.mean(cors_list)
                cnt = np.sum(cors_list)
                total_cnt = len(cors_list)
                log_str = f"Level {level} Accuracy = {cnt}/{total_cnt} = {acc:.3f}"
                f.write(log_str + "\n")

            f.write("\n===== Subject Accuracy =====\n")
            for subject in subjects:
                if subject in subject_cors:
                    cors_list = subject_cors[subject]
                    acc = np.mean(cors_list)
                    cnt = np.sum(cors_list)
                    total_cnt = len(cors_list)
                    log_str = f"{subject} Accuracy = {cnt}/{total_cnt} = {acc:.3f}"
                    f.write(log_str + "\n")

            f.write("\n===== Overall Result =====\n")
            final_log = f"Overall Accuracy = {correct_cnt}/{total} = {final_accuracy:.3f}\n"
            final_log += f"Skipped = {skipped}\n"
            final_log += f"Format Failed = {fmt_fail_cnt}\n"
            f.write(final_log)
        return

    dataset = load_dataset(args.data_path)['test']
    if args.eval_n > 0:
        dataset = dataset.select(range(min(args.eval_n, len(dataset))))

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if args.torch_dtype == "bf16" else torch.float16,
        device_map='cuda',
        trust_remote_code=True,
        use_cache=True
    )

    if args.load_adapter:
        model = PeftModel.from_pretrained(
            model,
            args.load_adapter,
            device_map='cuda',
            torch_dtype=torch.bfloat16 if args.torch_dtype == "bf16" else torch.float16
        )
        model = model.merge_and_unload()

    model.eval()

    total = 0
    correct_cnt = 0
    skipped = 0
    fmt_fail_cnt = 0

    cors = {}
    level_cors = {}
    subject_cors = {}
    subjects = ['Prealgebra', 'Algebra', 'Number Theory', 'Counting & Probability', 'Geometry', 'Intermediate Algebra',
                'Precalculus']

    jsonl_filename = "eval.jsonl"
    score_path = "eval_scores.txt"

    if args.load_adapter:
        try:
            parts = [p for p in args.load_adapter.split("/") if p]
            step = parts[-1].split("-")[-1]
            train_dir = next((p for p in parts if p.startswith("math_config_lr")), "")
            lr = train_dir.split("_lr")[-1].split("_t")[0]
            if lr and step:
                jsonl_filename = f"adapter_{lr}_{step}_{jsonl_filename}"
                score_path = f"{lr}_{step}_scores.txt"
        except Exception:
            pass

    jsonl_path = log_dir / jsonl_filename
    score_path = log_dir / score_path

    GENERATION_CONFIG = {
        "max_new_tokens": 768,
        "temperature": 0.0,
        "do_sample": False,
        "eos_token_id": None,
        "pad_token_id": None,
    }

    with tqdm(total=len(dataset), dynamic_ncols=True) as pbar:
        for idx, row in enumerate(dataset):
            prob_level, prob_type = get_level_type(row)
            gold_answer = row['answer']

            if gold_answer is None:
                skipped += 1
                pbar.update(1)
                continue

            user_content = f"{QUERY_TEMPLATE}\nQ: {row['problem'].strip()}\nA:"
            message = [
                {'role': 'user', 'content': user_content},
            ]

            if 'Qwen' in args.base_model:
                formatted_message = tokenizer.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
            else:
                formatted_message = tokenizer.apply_chat_template(
                    message,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            inputs = tokenizer(
                formatted_message,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=True
            ).to("cuda")

            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
                        temperature=GENERATION_CONFIG["temperature"],
                        do_sample=GENERATION_CONFIG["do_sample"],
                        top_p=1,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                generation = tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()
            except Exception as e:
                generation = ""
                skipped += 1
                pbar.update(1)
                continue

            pred_answer = last_boxed_only_string(generation)
            if pred_answer is None:
                skipped += 1
                fmt_fail_cnt += 1
                correct = False
            else:
                correct = is_equiv_by_api(pred_answer, gold_answer)
                if correct:
                    correct_cnt += 1

            total += 1

            if (prob_level, prob_type) in cors:
                cors[(prob_level, prob_type)].append(correct)
            else:
                cors[(prob_level, prob_type)] = [correct]

            if prob_level is not None:
                if prob_level in level_cors:
                    level_cors[prob_level].append(correct)
                else:
                    level_cors[prob_level] = [correct]

            if prob_type is not None:
                if prob_type in subject_cors:
                    subject_cors[prob_type].append(correct)
                else:
                    subject_cors[prob_type] = [correct]

            log_entry = {
                "id": idx,
                "problem": row['problem'],
                "generation": generation,
                "pred_answer": pred_answer,
                "gold_answer": gold_answer,
                "is_correct": correct,
                "level": prob_level,
                "type": prob_type,
                "format_failed": pred_answer is None,
                "skipped": pred_answer is None
            }
            import json
            with open(jsonl_path, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

            accuracy = 100 * correct_cnt / total if total > 0 else 0
            pbar.set_description(f"Accuracy: {accuracy:.2f}% | Skipped: {skipped} | Format Failed: {fmt_fail_cnt}")
            pbar.update(1)

    final_accuracy = correct_cnt / total if total > 0 else 0
    fmt_fail_rate = fmt_fail_cnt / total if total > 0 else 0
    skip_rate = skipped / (total + skipped) if (total + skipped) > 0 else 0

    with open(score_path, "w", encoding="utf-8") as f:
        f.write("===== Subject + Difficulty Accuracy =====\n")
        for prob_type in subjects:
            for prob_level in [1, 2, 3, 4, 5]:
                if (prob_level, prob_type) in cors:
                    cors_list = cors[(prob_level, prob_type)]
                    acc = np.mean(cors_list)
                    cnt = np.sum(cors_list)
                    total_cnt = len(cors_list)
                    log_str = f"{prob_type} Level {prob_level} Accuracy = {cnt}/{total_cnt} = {acc:.3f}"
                    f.write(log_str + "\n")

        f.write("\n===== Difficulty Accuracy =====\n")
        for level in sorted(level_cors):
            cors_list = level_cors[level]
            acc = np.mean(cors_list)
            cnt = np.sum(cors_list)
            total_cnt = len(cors_list)
            log_str = f"Level {level} Accuracy = {cnt}/{total_cnt} = {acc:.3f}"
            f.write(log_str + "\n")

        f.write("\n===== Subject Accuracy =====\n")
        for subject in subjects:
            if subject in subject_cors:
                cors_list = subject_cors[subject]
                acc = np.mean(cors_list)
                cnt = np.sum(cors_list)
                total_cnt = len(cors_list)
                log_str = f"{subject} Accuracy = {cnt}/{total_cnt} = {acc:.3f}"
                f.write(log_str + "\n")

        f.write("\n===== Overall Result =====\n")
        final_log = f"Overall Accuracy = {correct_cnt}/{total} = {final_accuracy:.3f}\n"
        final_log += f"Skipped = {skipped}\n"
        final_log += f"Format Failed = {fmt_fail_cnt}\n"
        f.write(final_log)

    del model, tokenizer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MATH dataset evaluation script (LaTeX-compatible)")
    parser.add_argument("--load_adapter", type=str, help="LoRA adapter path", default=None)
    parser.add_argument("--base_model", type=str, help="base model path/name", default="")
    parser.add_argument("--torch_dtype", type=str, choices=["bf16", "fp16"], help="model precision", default="bf16")
    parser.add_argument("--result_path", type=str,
                        default=None)
    parser.add_argument("--data_path", type=str,
                        default="",
                        help="MATH dataset path")
    parser.add_argument("--result_dir", type=str,
                        default="",
                        help="evaluation result output directory")
    parser.add_argument("--eval_n", type=int, default=0, help=">0 only evaluate first N test samples; 0=all")
    args = parser.parse_args()
    main(args)