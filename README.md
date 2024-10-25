# Investigating the Shortcomings of LLMs in Step-by-Step Legal Reasoning

### Instructions to obtain the dataset:
For obtaining permission to use the 'Civ. Pro.' dataset evaluated in our research paper, please refer and follow the steps as outlined in the following github link:
[url]([url](https://github.com/trusthlt/legal-argument-reasoning-task))

### Instructions to run the LLM inferences:
Please refer to the prompt-format in 'Legal-Reasoning-zero-shot-prompt-for-LLM-inference.py' for zero-shot CoT LLM inference

### Instructions to run auto-evaluator pipeline:
The project was tested using python 3.10.

Please use the following commands to setup the environment:
```
python -m venv legal_reasoning

# Activate the virtual environment (macOS/Linux)
source venv/bin/activate

# Activate the virtual environment (Windows): .\venv\Scripts\activate

# Install dependencies from requirements.txt
pip install -r requirements.txt
```

Run the notebooks 'Legal-Reasoning-Auto-Evaluator-Pipeline.ipynb' and 'Legal_Reasoning_Metrics_Calculation.ipynb' sequentially to run first run the auto-evaluator pipeline and then compute the 'Soundness', 'Correctness' and 'Accuracy' metrics as discussed in the paper.

### Instructions to run mitigation techniques:
Download Lllama models from [huggingface](https://huggingface.co/meta-llama).
Add your data path and the output path.

Run the baseline code as follows.

```python baseline.py```

To run other prompting techniques with error feedbacks, specify the feedback type from ```long```, ```short``` or  ```generic```.
As an example, you can run plan and solve technique with descriptive error feedbacks as follows.

```python plan-and-solve-w-errors.py --feedback long```

#### Additional Note:
There are various places in 'Legal-Reasoning-Auto-Evaluator-Pipeline' and 'Legal_Reasoning_Metrics_Calculation' where the paths of input and output files need to be provided for loading and saving of results. OpenAI API key is required to be provided in the appropriate place for the auto-evaluator pipeline to work.

