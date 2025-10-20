import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
from tqdm import tqdm

# Hugging Face Token
Token = ""

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=Token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    token=Token
)

with open("dataset/benchmark.json", "r") as f:
    data = json.load(f)

questions = data.get("thyroid", {})

rows = []
for qid, content in questions.items():
    options = content.get("options", {})
    row = {
        "id": qid,
        "topic": "thyroid",
        "question": content.get("question", ""),
        "option_A": options.get("A", ""),
        "option_B": options.get("B", ""),
        "option_C": options.get("C", ""),
        "option_D": options.get("D", ""),  # 如果缺失就填空
        "answer": content.get("answer", "")
    }
    rows.append(row)
df = pd.DataFrame(rows)


def build_prompt(row):
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
You are a licensed medical doctor. Based on the patient's case, choose the most likely diagnosis from the options.

Question:
{row['question']}

Options:
A. {row['option_A']}
B. {row['option_B']}
C. {row['option_C']}
D. {row['option_D']}

Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    return prompt

def extract_answer(output_text):
    match = re.search(r'\b([A-D])\b', output_text)
    if match:
        return match.group(1).upper()
    return None

predictions = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    prompt = build_prompt(row)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False
    )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pred = extract_answer(output_text)
    predictions.append(pred)

df["prediction"] = predictions
df["correct"] = df["prediction"] == df["answer"]

df.to_csv("medical_mcq_predictions.csv", index=False)

accuracy = df["correct"].mean()
print(f"Accuracy: {accuracy:.2%}")

