

from datasets import load_dataset
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import transformers
import datasets
from transformers import AutoTokenizer, TFPegasusForConditionalGeneration, TFLEDForConditionalGeneration
import datetime
import os

data_dir = "./data"
log_dir = f"{data_dir}/experiments/Pegasus-sum/logs"
save_path = f"{data_dir}/experiments/Pegasus-sum/models"
cache_path_train = f"{data_dir}/cache/Pegasus-gigaword.train"
cache_path_test = f"{data_dir}/cache/Pegasus-gigaword.test"
cache_path_val = f"{data_dir}/cache/Pegasus-gigaword.val"

save_freq = 256
device = '/GPU:0' # tensorflow style device
class custom_model_save(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.idx = 0
    def on_batch_begin(self, epoch, logs=None):
        self.idx += 1
        if self.idx % save_freq == 0:
            self.model.save_pretrained(save_path)
class SummarizePegasus(TFPegasusForConditionalGeneration):
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


tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")



train_dataset = load_dataset('gigaword', split='train[:100000]')
valid_dataset = load_dataset('gigaword', split='validation[:20000]')

train_dataset.features

data = next(iter(train_dataset))
print("Example data from the dataset: \n", data)

warmup_steps = 300
batch_size = 16
encoder_max_len = 1024
decoder_max_len = 128
buffer_size = 1000
ntrain = len(train_dataset)
nvalid = len(valid_dataset)
steps = int(np.ceil(ntrain/batch_size))
valid_steps = int(np.ceil(nvalid/batch_size))
print("Total Steps: ", steps)
print("Total Validation Steps: ", valid_steps)


def encode(example,
           encoder_max_len=encoder_max_len, decoder_max_len=decoder_max_len):

    context = example['document']
    answer = example['summary']

    question_plus = f"Summarize: {str(context)} </s>"

    
    answer_plus = f"{answer} </s>"

    encoder_inputs = tokenizer(question_plus, truncation=True,
                               return_tensors='tf', max_length=encoder_max_len,
                              pad_to_max_length=True)

    decoder_inputs = tokenizer(answer_plus, truncation=True,
                               return_tensors='tf', max_length=decoder_max_len,
                              pad_to_max_length=True)

    input_ids = encoder_inputs['input_ids'][0]
    input_attention = encoder_inputs['attention_mask'][0]
    target_ids = decoder_inputs['input_ids'][0]
    target_attention = decoder_inputs['attention_mask'][0]

    outputs = {'input_ids':input_ids, 'attention_mask': input_attention,
               'labels':target_ids, 'decoder_attention_mask':target_attention}
    return outputs



train_ds = train_dataset.map(encode, cache_file_name=cache_path_train, load_from_cache_file=True)
valid_ds = valid_dataset.map(encode, cache_file_name=cache_path_val, load_from_cache_file=True)


ex = next(iter(train_ds))
print("Example data from the mapped dataset: \n", ex)



def to_tf_dataset(dataset):
  columns = ['input_ids', 'attention_mask', 'labels', 'decoder_attention_mask']
  dataset.set_format(type='tensorflow', columns=columns)
  return_types = {'input_ids':tf.int32, 'attention_mask':tf.int32,
                'labels':tf.int32, 'decoder_attention_mask':tf.int32,  }
  return_shapes = {'input_ids': tf.TensorShape([None]), 'attention_mask': tf.TensorShape([None]),
                  'labels': tf.TensorShape([None]), 'decoder_attention_mask':tf.TensorShape([None])}
  ds = tf.data.Dataset.from_generator(lambda : dataset, return_types, return_shapes)
  return ds



tf_train_ds = to_tf_dataset(train_ds)
tf_valid_ds = to_tf_dataset(valid_ds)


def create_dataset(dataset, cache_path=None, batch_size=4,
                   buffer_size= 1000, shuffling=True):
    if cache_path is not None:
        dataset = dataset.cache(cache_path)
    if shuffling:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset



tf_train_ds= create_dataset(tf_train_ds, batch_size=batch_size,
                         shuffling=True, cache_path = None)
tf_valid_ds = create_dataset(tf_valid_ds, batch_size=batch_size,
                         shuffling=False, cache_path = None)



class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, warmup_steps=1e4):
    super().__init__()

    self.warmup_steps = tf.cast(warmup_steps, tf.float32)

  def __call__(self, step):
    step = tf.cast(step, tf.float32)
    m = tf.maximum(self.warmup_steps, step)
    m = tf.cast(m, tf.float32)
    lr = tf.math.rsqrt(m)

    return lr

schedule = CustomSchedule()

start_profile_batch = steps+10
stop_profile_batch = start_profile_batch + 100
profile_range = f"{start_profile_batch},{stop_profile_batch}"

log_path = log_dir + "/" + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1,
                                                     update_freq=20,profile_batch=profile_range)

checkpoint_filepath = save_path + "/" + "Pegasus-sum-checkpoint.h5"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    mode='min',
    save_freq = 8)

callbacks = [tensorboard_callback, custom_model_save()]
metrics = [tf.keras.metrics.SparseTopKCategoricalAccuracy(name='accuracy') ]

#learning_rate = CustomSchedule()
learning_rate = 0.0005  # Instead set a static learning rate
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate)


with tf.device(device):
    model = SummarizePegasus.from_pretrained("google/pegasus-xsum")
    model.load_weights(save_path + '/tf_model.h5')

with tf.device(device):
    model.compile(optimizer=optimizer, metrics=metrics)

with tf.device(device):
    epochs_done = 0
    model.fit(tf_train_ds, epochs=10, steps_per_epoch=steps, callbacks=callbacks,
            validation_data=tf_valid_ds, validation_steps=valid_steps, initial_epoch=epochs_done)

model.save_pretrained(save_path)

with tf.device(device):
    print("starting model load")
    model = SummarizePegasus.from_pretrained("google/pegasus-xsum")
    print("finished base load")
    model.load_weights(save_path + '/tf_model.h5')
    print("finished loading model")

context = """I played chess with one of my co-workers at work. I played during my break. I won the game. We had a few people watching us play. They all supposedly funny stories about how they know chess but they don't"""

print(context)

input_text =  f"Summarize: {str(context)} </s>"
encoded_query = tokenizer(input_text,
                         return_tensors='tf', pad_to_max_length=True, truncation=True, max_length=encoder_max_len)
input_ids = encoded_query["input_ids"]
attention_mask = encoded_query["attention_mask"]
generated_answer = model.generate(input_ids, attention_mask=attention_mask,
                                 max_length=decoder_max_len, top_p=0.95, top_k=500, repetition_penalty=2.0)
decoded_answer = tokenizer.decode(generated_answer.numpy()[0])
print("Answer: ", decoded_answer)







