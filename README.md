# Investigating the Shortcomings of LLMs in Step-by-Step Legal Reasoning

### Instructions to obtain the dataset:
For obtaining permission to use the 'Civ. Pro.' dataset evaluated in our research paper, please refer and follow the steps as outlined in the following github link:
[https://github.com/trusthlt/legal-argument-reasoning-task](url)

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

