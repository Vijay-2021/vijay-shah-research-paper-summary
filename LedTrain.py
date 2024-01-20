
#!/usr/bin/env python
# coding: utf-8

#  Imports

import time
import torch
from transformers import LEDTokenizer, LEDForConditionalGeneration, AutoTokenizer, LEDForConditionalGeneration
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from tqdm.auto import tqdm
from accelerate import Accelerator
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler, AutoModel,AutoModelForSeq2SeqLM
import utils.pdf_reader as pdfr
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from datasets import load_dataset, load_metric
import evaluate
import gc
from torch.distributed.fsdp import (
   FullyShardedDataParallel,
   CPUOffload,
)


accelerator = Accelerator(log_with="all")
print("~~~~~~~~~~~~~Intialized Accelerator~~~~~~~~~~~~~~~~~~~~\n\n\n")

print("~~~~~~~~~~~~~Finished Imports~~~~~~~~~~~~~~~~~~~~\n\n\n")
device = "cuda"

data_dir = "./data"
log_dir = f"{data_dir}/experiments/LED/logs"
save_path = f"{data_dir}/experiments/LED/models/output2.pt"
cache_path_train = f"{data_dir}/cache/Narrative.train"
cache_path_val = f"{data_dir}/cache/Narrative.val"
cache_path_test = f"{data_dir}/cache/Narrative.test"

tokenizer = AutoTokenizer.from_pretrained("pszemraj/led-large-book-summary")
train_dataset = load_dataset("narrativeqa", split="train[:3000]")
val_dataset = load_dataset("narrativeqa", split="validation[:1000]") # run some tests on a smaller sample size for now

encoder_max_len = 16384
decoder_max_len = 512
batch_size = 1

def encode(example,
           encoder_max_len=encoder_max_len, decoder_max_len=decoder_max_len):
  
    context = example['document']['text']
    question = example['question']['text']
    answer = example['answers'][0]['text']
  
    question_plus = f"answer_me: {str(question)}"
    question_plus += f" context: {str(context)} </s>"
    
    answer_plus = ', '.join([i for i in list(answer)])
    answer_plus = f"{answer_plus} </s>"
    
    encoder_inputs = tokenizer(question_plus, truncation=True, 
                               return_tensors='pt', max_length=encoder_max_len,
                              pad_to_max_length=True)
    
    decoder_inputs = tokenizer(answer_plus, truncation=True, 
                               return_tensors='pt', max_length=decoder_max_len,
                              pad_to_max_length=True)
    
    input_ids = encoder_inputs['input_ids'][0]
    input_attention = encoder_inputs['attention_mask'][0]
    target_ids = decoder_inputs['input_ids'][0]
    target_attention = decoder_inputs['attention_mask'][0]
    global_attention = torch.ones_like(input_attention)
    outputs = {'input_ids':input_ids, 'attention_mask': input_attention, 
               'labels':target_ids, 'decoder_attention_mask':target_attention, 'global_attention_mask':global_attention}
    return outputs



narrative_train = train_dataset.map(encode, cache_file_name=cache_path_train, load_from_cache_file=True)
narrative_val = val_dataset.map(encode, cache_file_name=cache_path_val, load_from_cache_file=True)

def to_dataset(dataset):  
  columns = ['input_ids', 'attention_mask', 'labels', 'decoder_attention_mask']
  dataset.set_format(type='torch', columns=columns)
  return dataset
  
train_ds = to_dataset(narrative_train)
valid_ds = to_dataset(narrative_val)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader = DataLoader(
    train_ds, shuffle=True, batch_size=batch_size, collate_fn=data_collator
)

val_dataloader = DataLoader(
    valid_ds, batch_size=batch_size, collate_fn=data_collator
)

print("~~~~~~~~~~~~~Finished Dataset Loading~~~~~~~~~~~~~~~~~~~~\n\n\n")

gc.collect()
torch.cuda.empty_cache()


model = AutoModelForSeq2SeqLM.from_pretrained("pszemraj/led-large-book-summary", offload_folder="offload", torch_dtype=torch.float16).half().cuda()
optimizer = AdamW(model.parameters(), lr=1e-5)

num_epochs = 5

num_training_steps = num_epochs * len(train_dataloader)
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)



print("~~~~~~~~~~~~~Loaded Model and Optimizer~~~~~~~~~~~~~~~~~~~~\n\n\n")

device_map = [True, True, True, True, True]


print("~~~~~~~~~~~~~Prepared Accelerator~~~~~~~~~~~~~~~~~~~~\n\n\n")


num_epochs = 5
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)


progress_bar = tqdm(range(num_training_steps))

print("~~~~~~~~~~~~~Finished Setup~~~~~~~~~~~~~~~~~~~~\n\n\n")


iter = 0
metric = evaluate.load("rouge")

for epoch in range(num_epochs):
    print("starting epoch\n")
    model.train()
    for batch in train_dataloader:
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        if iter % 100 == 0:
            model.save_pretrained(save_path)
            print("saving model state")
            print("loss is: ", loss)
        iter += 1
    model.eval()

               

context = """We went on a trip to Europe. We had our breakfast at 7 am in the morning at \
the nearby coffee shop. Wore a dark blue over coat for our first visit to Louvre Museum \
to experience history and art."""

question = "When was breakfast?"
print(context)
print(question)



input_text =  f"answer_me: {question} context: {context} </s>"
encoded_query = tokenizer(input_text, 
                         return_tensors='pt', pad_to_max_length=True, truncation=True, max_length=encoder_max_len).to(device)
input_ids = encoded_query["input_ids"].to(device)
attention_mask = encoded_query["attention_mask"].to(device)
global_attention_mask = torch.ones_like(attention_mask).to(device)
generated_answer = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask,
                                 max_length=decoder_max_len, top_k=50, repetition_penalty=2.0)
decoded_answer = tokenizer.decode(generated_answer.cpu().numpy()[0])
print("Answer: ", decoded_answer)







