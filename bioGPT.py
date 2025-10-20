import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model
model_name = "microsoft/BioGPT-Large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def answer_mcq(question, options, few_shot_examples=None):
    prompt = ""
    if few_shot_examples:
        for ex in few_shot_examples:
            prompt += f"Q: {ex['question']}\n"
            for k, v in ex["options"].items():
                prompt += f"{k}. {v}\n"
            prompt += f"Answer: {ex['answer']}\n\n"

    prompt += f"Q: {question}\n"
    for k, v in options.items():
        prompt += f"{k}. {v}\n"
    prompt += "Answer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    def extract_answer(text):
        answer_lines = [line for line in text.split("\n") if "Answer" in line]
        if answer_lines:
            match = re.search(r"\b([ABCD])\b", answer_lines[-1])
            if match:
                return match.group(1)
        return None

    predicted = extract_answer(text)
    return predicted, text


#  Read JSON files
with open("dataset/benchmark.json", "r", encoding="utf-8") as f:
    data = json.load(f)

questions_dict = data.get("thyroid", {})

correct = 0
total = 0
for qid, qdata in questions_dict.items():
    question = qdata["question"]
    options = qdata["options"]
    answer = qdata["answer"]

    predicted, raw_output = answer_mcq(question, options)

    print(f"Question ID: {qid}")
    print(f"Predicted: {predicted} | True: {answer}")
    print("-----")
    total += 1
    if predicted == answer:
        correct += 1

accuracy = correct / total if total > 0 else 0
print(f"Accuracy: {accuracy*100:.2f}%")

