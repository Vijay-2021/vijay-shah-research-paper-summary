

from datasets import load_dataset
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import transformers
import datasets
from transformers import AutoTokenizer, TFT5ForConditionalGeneration, TFLEDForConditionalGeneration
import datetime
import os

data_dir = "./data"
log_dir = f"{data_dir}/experiments/t5-sum/logs"
save_path = f"{data_dir}/experiments/t5-sum/models"
cache_path_train = f"{data_dir}/cache/t5-sum.train"
cache_path_test = f"{data_dir}/cache/t5-sum.test"
cache_path_val = f"{data_dir}/cache/t5-sum.val"

save_freq = 32
device = '/GPU:0' # tensorflow style device
class custom_model_save(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.idx = 0
    def on_batch_begin(self, epoch, logs=None):
        self.idx += 1
        if self.idx % save_freq == 0:
            print("saving model!!!")
            self.model.save_pretrained(save_path)
class SummarizeT5(TFT5ForConditionalGeneration):
    def __init__(self, *args, log_dir=None, cache_dir= None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker= tf.keras.metrics.Mean(name='loss')
    @tf.function
    def train_step(self, data):
        x = data
        y = x["labels"]
        y = tf.reshape(y, [-1, 1])
        with tf.GradientTape() as tape:
            outputs = self(x, training=True)
            loss = outputs[0]
            logits = outputs[1]
            loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        lr = self.optimizer._decayed_lr(tf.float32)

        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, logits)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics.update({'lr': lr})

        return metrics

    def test_step(self, data):
        x = data
        y = x["labels"]
        y = tf.reshape(y, [-1, 1])
        output = self(x, training=False)
        loss = output[0]
        loss = tf.reduce_mean(loss)
        logits = output[1]

        self.loss_tracker.update_state(loss)
        self.compiled_metrics.update_state(y, logits)
        return {m.name: m.result() for m in self.metrics}


tokenizer = AutoTokenizer.from_pretrained("t5-base")




warmup_steps = 300
batch_size = 16
encoder_max_len = 512
decoder_max_len = 54
buffer_size = 1000



with tf.device(device):
    print("starting model load")
    model = SummarizeT5.from_pretrained("t5-base")
    print("finished base load")
    model.load_weights(save_path + '/tf_model.h5')
    print("finished loading model")

context = """I played chess with one of my co-workers at work. I played during my break. I won the game. We had a few people watching us play. They all supposedly funny stories about how they know chess but they don't"""

question = "When did I play the game?"
print(context)
print(question)

input_text =  f"Summarize: {str(context)} </s>"
encoded_query = tokenizer(input_text,
                         return_tensors='tf', pad_to_max_length=True, truncation=True, max_length=encoder_max_len)
input_ids = encoded_query["input_ids"]
attention_mask = encoded_query["attention_mask"]
generated_answer = model.generate(input_ids, attention_mask=attention_mask,
                                 max_length=decoder_max_len, top_p=0.95, top_k=500, repetition_penalty=2.0)
decoded_answer = tokenizer.decode(generated_answer.numpy()[0])
print("Answer: ", decoded_answer)







