from datetime import datetime
from tqdm import tqdm
import os
import pandas as pd

import transformers
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={
      "torch_dtype": torch.bfloat16
      },
    device_map="auto",
)

df = pd.read_csv('LEGAL_REASONING_DATASET.csv') # add the path to the dataset

current_date_time = datetime.now()
formatted_date_time = current_date_time.strftime("%m-%d-%H-%M")

exp_name = 'OUTPUT_PATH' + formatted_date_time # add the path to the output directory
os.makedirs(exp_name, exist_ok=True)

for i in tqdm(range(175)): # run for the all whole dataset

    prompt = "Legal Context:\n" + df['Context'][i] + "\n" \
    "Question:\n" + df['Question'][i] + "\n\n" \
    + df['Options'][i] \

    prompt = prompt + "\n\nProvide the response in the following format: \nExplanation: [Legal reasoning step by step as numbered points]\nFinal answer: [Final answer as an English capital letter from the options given above]"

    conversation = [
            {"role": "system", "content": "You are an expert legal assistant."},
            {"role": "user", "content": prompt}
        ]

    outputs = pipeline(
        conversation,
        max_new_tokens=4096,
        temperature=0.001
    )

    final_output = outputs[0]["generated_text"][-1]['content']

    with open(exp_name + '/response_' + str(i+1) + '.txt', 'w') as f:
        f.write(final_output)