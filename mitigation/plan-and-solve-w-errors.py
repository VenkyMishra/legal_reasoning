from datetime import datetime
from tqdm import tqdm
import os
import pandas as pd
import argparse

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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback", type=str, default="long", help="Feedback type: long, short, generic")
    args = parser.parse_args()
    return args

def main():

    args = get_args()
    feedback_type = args.feedback

    with open(f"feedback_{feedback_type}.txt", "r") as f:
        system_prompt = f.read()

    df = pd.read_csv('LEGAL_REASONING_DATASET.csv') # add the path to the dataset

    current_date_time = datetime.now()
    formatted_date_time = current_date_time.strftime("%m-%d-%H-%M")

    exp_name = f'OUTPUT_PATH' + formatted_date_time # add the path to the output directory
    os.makedirs(exp_name, exist_ok=True)

    print("*********** Starting the experiment ***********")
    print("Feedback type: ", feedback_type)
    print('System prompt: ', system_prompt)
    print("************************************************")

    for i in tqdm(range(175)): # run for the all whole dataset

        prompt = "Legal Context:\n" + df['Context'][i] + "\n" \
        "Question:\n" + df['Question'][i] + "\n\n" \
        + df['Options'][i] \

        prompt1 = prompt + "\n\nLet's first understand the problem and devise a plan to solve the problem. Do not solve the problem."

        conversation = [
                {"role": "system", "content": "You are an expert legal assistant."},
                {"role": "user", "content": prompt1}
            ]

        plan = pipeline(
            conversation,
            max_new_tokens=4096,
            temperature=0.001
        )[0]["generated_text"][-1]['content']

        prompt2 = f"Let's carry out the plan and solve the problem step-by-step.\n\n{prompt}\n\nPlan:\n{plan}\n\nAt the end provide final answer as Final answer: [Final answer as an English capital letter from the options given above]"
    
        conversation2 = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt2}
        ]
        
        final_output = pipeline(
            conversation2,
            max_new_tokens=4096,
            temperature=0.001
        )[0]["generated_text"][-1]['content']

        with open(exp_name + '/response_' + str(i+1) + '.txt', 'w') as f:
            f.write(final_output)

if __name__ == "__main__":
    main()

# python plan-and-solve-w-errors.py --feedback long