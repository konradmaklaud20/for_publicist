import pandas as pd
import re
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, GPT2LMHeadModel
from transformers import pipeline
import nltk


# pip install git+git://github.com/huggingface/transformers/

data = pd.read_csv('essays.csv')  # Файл с текстами различных очерков, статей, эссе

nl = []
for i in data['text']:
    nl.append(i)


def build_text_files(txts, dest_path):
    f = open(dest_path, 'w')
    data = ''
    for txt in txts:
        summary = txt.strip()
        summary = re.sub(r"\s", " ", summary)
        data += summary + "  "
    f.write(data)


train, test = train_test_split(nl, test_size=0.15)
build_text_files(train,'train_dataset.txt')
build_text_files(test,'test_dataset.txt')

print("Train dataset length: " + str(len(train)))
print("Test dataset length: " + str(len(test)))

tokenizer = GPT2Tokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')

train_path = 'train_dataset.txt'
test_path = 'test_dataset.txt'


def load_dataset(train_path,test_path,tokenizer):

    train_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size=16)

    test_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=test_path,
          block_size=16)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,

    )
    return train_dataset,test_dataset,data_collator


train_dataset, test_dataset, data_collator = load_dataset(train_path, test_path, tokenizer)


model = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")

training_args = TrainingArguments(
    output_dir="./gpt2",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    eval_steps=400,
    save_steps=800,
    warmup_steps=500,
    )

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,

)

trainer.train()
trainer.save_model()

text_pipeline = pipeline('text-generation', model='./gpt2', tokenizer='sberbank-ai/rugpt3medium_based_on_gpt2',
                config={'max_length':1000, 'temperature': .5})

initial_seed = "Я думаю, что смысл жизни в"

initial_seed_split = initial_seed.split()

if len(initial_seed_split) > 20:
    initial_seed_split = initial_seed_split[len(initial_seed_split) - 20:]

t = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
text = ' '.join(initial_seed_split)
length = len(t(text)['input_ids'])

speech = text_pipeline(text, max_length=length + 20, min_length=length + 5,
               early_stopping=True, do_sample=True, temperature=1.2)[0]['generated_text']

speech = speech.replace(text, '')
speech_tokens = nltk.sent_tokenize(speech)
speech_text = speech_tokens[0]

# Выводим только завершённые предложения
if len(speech_tokens) > 2:
    speech_text = ' '.join(speech_tokens[:2])

# Очищаем текст от двойных пробелов, незакрытых кавычек и скобок
speech_text = re.sub(r'\s+', ' ', speech_text)

if speech_text.count('"') % 2 != 0:
    speech_text = speech_text.replace('"', '')
if speech_text.count(')') % 2 != 0:
    speech_text = speech_text.replace(')', '')
if speech_text.count('(') % 2 != 0:
    speech_text = speech_text.replace('(', '')

print(speech_text)
