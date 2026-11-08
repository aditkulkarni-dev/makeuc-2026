import json
import os
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration
MODEL_ID = "google/gemma-2-2b-it"
DATASET_ID = "MatanBT/gcg-evaluated-data"

MAX_TESTS = 100  # Start with 20; set to None for all rows
MAX_NEW_TOKENS = 160

RESULTS_FILE = "results/baseline_results.jsonl"

os.makedirs("results", exist_ok=True)

# Load Gemma only once
print("Loading Gemma...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="cuda"
)

model.eval()

# Stream the dataset instead of loading everything into RAM
print("Loading dataset...")

dataset = load_dataset(
    DATASET_ID,
    split="gemma2",
    streaming=True
)

# Resume from the last saved row if interrupted
completed = set()

if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            completed.add(record["row_index"])

print("Starting baseline evaluation...")

tested = 0

with open(RESULTS_FILE, "a", encoding="utf-8") as output:

    for row_index, row in enumerate(dataset):

        if row_index in completed:
            continue

        if MAX_TESTS is not None and tested >= MAX_TESTS:
            break

        # Use the dataset's exact combined prompt
        prompt = row["message_suffixed"]

        messages = [
            {"role": "user", "content": prompt}
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to("cuda")

        # No refusal vector or steering hook applied
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        new_tokens = generated[
            0, inputs["input_ids"].shape[1]:
        ]

        response = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
        )

        record = {
            "row_index": row_index,
            "message_id": row["message_id"],
            "suffix_id": row["suffix_id"],
            "prompt": prompt,
            "response": response,
            "published_score": row["strongreject_finetuned"],
            "published_category": row["response_category"],
            "steering_strength": 0.0
        }

        output.write(json.dumps(record, ensure_ascii=False) + "\n")
        output.flush()

        tested += 1

        print(
            f"Completed {tested} | "
            f"Dataset row {row_index} | "
            f"Published score: {row['strongreject_finetuned']:.3f}"
        )

print("Evaluation finished.")
print("Results saved to:", RESULTS_FILE)
