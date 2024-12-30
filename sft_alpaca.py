from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig
from torch.utils.data import DataLoader
import torch
import time

TOTAL_BATCH_SIZE = 128
BATCH_SIZE = 2
LR = 5e-5
EPOCHS = 3
EVAL_EVERY = 100
VAL_STEPS = 64

model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.model_max_length = 512

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,  bnb_4bit_quant_type="nf4")
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map="auto")

lora_config = LoraConfig(r=64, lora_alpha=16, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")

model = get_peft_model(model, lora_config)

dataset = load_dataset("tatsu-lab/alpaca")

def format_tokenize(examples):
    chats = [
        [
            {"role": "user", "content": instruction + (f"\n{input}" if input else "")},
            {"role": "assistant", "content": output}
        ]
        for instruction, input, output in zip(examples["instruction"], examples["input"], examples["output"])
    ]

    formatted = [tokenizer.apply_chat_template(chat, tokenize=False) for chat in chats]
    cols = tokenizer(formatted, padding="max_length", truncation=True, return_tensors="pt")
    cols['labels'] = cols['input_ids'].clone()
    return cols

tokenized_dataset = dataset["train"].map(format_tokenize, batched=True, remove_columns=["instruction","input", "output", "text"])
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)

train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=BATCH_SIZE, collate_fn=default_data_collator)
val_dataloader = DataLoader(tokenized_dataset["test"], batch_size=BATCH_SIZE, collate_fn=default_data_collator)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

grad_accum_steps = TOTAL_BATCH_SIZE // BATCH_SIZE
num_training_steps = (len(train_dataloader) // grad_accum_steps) * EPOCHS
steps_per_epoch = len(train_dataloader) // grad_accum_steps
num_warmup_steps = num_training_steps * 0.01
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

best_val_loss = float("inf")
for epoch in range(EPOCHS):
    model.train()
    for step in range(steps_per_epoch):
        if step > 0 and step % EVAL_EVERY == 0:
            model.eval()
            val_loss = 0
            for _ in range(VAL_STEPS):
                batch = next(iter(val_dataloader))
                batch = {k: v.to(model.device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)
                val_loss += outputs.loss.item()
            val_loss = val_loss / VAL_STEPS
            print(f"Validation Loss: {val_loss}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_pretrained("qwen2.5-0.5b-alpaca")
        
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

tokenizer.save_pretrained("qwen2.5-0.5b-alpaca")