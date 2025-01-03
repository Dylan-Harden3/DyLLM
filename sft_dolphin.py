from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig
from torch.utils.data import DataLoader
import torch
import time
import argparse

def train(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.model_max_length = args.max_length

    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,  bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, quantization_config=quantization_config, device_map="auto")

    lora_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.2, bias="none", task_type="CAUSAL_LM")

    model = get_peft_model(model, lora_config)

    dataset = load_dataset("cognitivecomputations/dolphin", data_files="flan1m-alpaca-uncensored.jsonl")

    def format_tokenize(examples):
        chats = [
            [
                {"role": "system", "content": instruction},
                {"role": "user", "content": input},
                {"role": "assistant", "content": output}
            ]
            for instruction, input, output in zip(examples["instruction"], examples["input"], examples["output"])
        ]
        formatted = [tokenizer.apply_chat_template(chat, tokenize=False) for chat in chats]
        cols = tokenizer(formatted, padding="max_length", truncation=True, return_tensors="pt")
        cols['labels'] = cols['input_ids'].clone()
        return cols

    tokenized_dataset = dataset["train"].map(format_tokenize, batched=True, remove_columns=["instruction", "input", "output"])
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)

    train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=args.batch_size, collate_fn=default_data_collator)
    val_dataloader = DataLoader(tokenized_dataset["test"], batch_size=args.batch_size, collate_fn=default_data_collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    grad_accum_steps = args.total_batch_size // args.batch_size
    num_training_steps = (len(train_dataloader) // grad_accum_steps) * args.epochs
    steps_per_epoch = len(train_dataloader) // grad_accum_steps
    num_warmup_steps = num_training_steps * 0.01
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    no_improve = 0

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        for step in range(steps_per_epoch):
            if step > num_warmup_steps and step % args.eval_every == 0:
                model.eval()
                val_loss = 0
                for _ in range(args.val_steps):
                    batch = next(iter(val_dataloader))
                    batch = {k: v.to(model.device) for k, v in batch.items()}
                    with torch.no_grad():
                        outputs = model(**batch)
                    val_loss += outputs.loss.item()
                val_loss = val_loss / args.val_steps
                print(f"Validation Loss: {val_loss}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model.save_pretrained(args.output_dir)
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve == 5:
                        print("No improvement in 5 validation evals, stopping training.")
                        break
            model.train()
            optimizer.zero_grad()
            train_loss = 0.0
            start = time.time()
        
            for _ in range(grad_accum_steps):
                batch = next(iter(train_dataloader))
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss / grad_accum_steps
                train_loss += loss.item()
                loss.backward()
            
            end = time.time()
            print(f"Step: {step + epoch * steps_per_epoch:4d}/{num_training_steps} | Loss: {train_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e} | Time: {end - start:.2f}")        
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)
            optimizer.step()
            scheduler.step()

    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output_dir", type=str, default="qwen2.5-0.5b-dolphin-2")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--total_batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--val_steps", type=int, default=64)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--clip_value", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.1)


    args = parser.parse_args()
    train(args)