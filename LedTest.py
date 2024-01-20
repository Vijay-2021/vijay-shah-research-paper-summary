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

device = "cuda"

data_dir = "./data"
log_dir = f"{data_dir}/experiments/LED/logs"
save_path = f"{data_dir}/experiments/LED/models/output2.pt"
cache_path_train = f"{data_dir}/cache/Narrative.train"
cache_path_val = f"{data_dir}/cache/Narrative.val"
cache_path_test = f"{data_dir}/cache/Narrative.test"

tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
train_dataset = load_dataset("narrativeqa", split="train[:3000]")
val_dataset = load_dataset("narrativeqa", split="validation[:1000]") # run some tests on a smaller sample size for now

encoder_max_len = 16384
decoder_max_len = 512
batch_size = 1
print("~~~~~~~~~~~~~~~~~~~~~~~~~~Starting Model Load~~~~~~~~~~~~~~~~~~~~\n\n\n")
model = AutoModelForSeq2SeqLM.from_pretrained(save_path,  offload_folder="offload", torch_dtype=torch.float16).half().cuda()
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