# -*- coding: utf-8 -*-
import torch
print("CUDA available: ", torch.cuda.is_available())
!nvidia-smi

!pip -q install transformers datasets evaluate accelerate --upgrade

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

from datasets import load_dataset
from transformers import AutoTokenizer

ckpt = "bert-base-uncased" # base BERT
ds = load_dataset("glue", "sst2") # SST-2 sentiment dataset
tok = AutoTokenizer.from_pretrained(ckpt)

def tokenize(batch):
  return tok(batch["sentence"], truncation=True, max_length=128)

ds_tok = ds.map(tokenize, batched=True, remove_columns=["sentence"])

from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
import numpy as np, evaluate

collator = DataCollatorWithPadding(tokenizer=tok)
model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2)

metric = evaluate.load("glue", "sst2")
def compute_metrics(eval_pred):
  logits, labels = eval_pred
  preds = np.argmax(logits, axis=-1)
  return metric.compute(predictions=preds, references=labels)
args = TrainingArguments(
    output_dir="bert-sst2-t4",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    logging_steps=50,
    do_eval=True,
    save_steps=500,
    eval_steps=500,
    report_to="none",
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_tok["train"],
    eval_dataset=ds_tok["validation"],
    tokenizer=tok,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

train_result = trainer.train()
train_result

metrics = trainer.evaluate()
metrics

from transformers import TextClassificationPipeline
pipe = TextClassificationPipeline(model=trainer.model, tokenizer=tok, device=0)
print(pipe("This movie was absolutely fantastic!"))
print(pipe("That was the worst meal I've had in years."))
