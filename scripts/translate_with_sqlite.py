import logging
import argparse
import json
import sqlite3
import torch
import nltk
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import load_llm, load_tokenizer

logger = logging.getLogger("translate_document")

def make_input_example(prompt, text):
    # Add the new input text to translate
    return prompt.replace("INPUT_TEXT", text)

def strip_prompt_from_decoded(prompt: str, generated_text: str) -> str:
    idx = generated_text.find(prompt)
    if idx != -1:
        return generated_text[idx + len(prompt):].lstrip()
    else:
        print("DIDN'T FIND PROMPT")
        # fallback: return whole if prompt isn't found (or log)
        return generated_text.strip()


def create_db(path):
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS translations (
            location TEXT PRIMARY KEY,
            source TEXT,
            output TEXT
        )
    """)
    conn.commit()
    return conn

def is_translated(cursor, location):
    cursor.execute("SELECT 1 FROM translations WHERE location = ?", (location,))
    return cursor.fetchone() is not None

def save_translations(cursor, rows):
    cursor.executemany(
        "INSERT OR IGNORE INTO translations (location, source, output) VALUES (?, ?, ?)", rows
    )

def main(args):

    tokenizer = load_tokenizer(args.model)
    model = load_llm(args.model)
  

    with open(args.prompt, 'rt') as f:
        prompt = f.read()

    conn = create_db(args.output)
    cursor = conn.cursor()

    with open(args.input, 'rt') as f:
        all_lines = [(str(json.loads(line)["location"]), json.loads(line)["text"]) for line in f]

    already_translated = set(
        row[0] for row in cursor.execute("SELECT location FROM translations").fetchall()
    )
    remaining_lines = [(loc, txt) for loc, txt in all_lines if loc not in already_translated]

    print(f"Total lines: {len(all_lines)} | Already translated: {len(already_translated)} | Lines to translate: {len(all_lines) - len(already_translated)}")

    for i in tqdm(range(0, len(remaining_lines), args.batch_size)):
        batch = remaining_lines[i:i + args.batch_size]
        input_texts = [make_input_example(prompt, txt) for _, txt in batch]

        try:
            encoded = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=100,
                    num_return_sequences=1,
                    repetition_penalty=1.1
                )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            stripped_decoded = [strip_prompt_from_decoded(input_texts[i], out) for i, out in enumerate(decoded)]

            results = [(loc, src, trans) for (loc, src), trans in zip(batch, stripped_decoded)]

            save_translations(cursor, results)
            conn.commit()
            del encoded, generated
            torch.cuda.empty_cache()

        except RuntimeError as e:
            print(f"[OOM or error]: {e}")
            torch.cuda.empty_cache()
            continue
        # sanity check
        if i > 1000:
            print(f"Processed {i} lines, stopping early for sanity check.")
            break

    conn.close()
    print("✅ Translation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="SQLite DB file to write")
    parser.add_argument("--prompt", required=True, help="Prompt file path")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    main(args)
