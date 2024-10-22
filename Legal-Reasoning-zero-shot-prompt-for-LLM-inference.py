system_prompt = "You are a helpful legal assistant. Choose the correct option by performing legal reasoning while strictly adhering to the legal context below." # The system-prompt
legal_context = "" # Legal-Context Extracted from the 'Civ-Pro' dataset
question = "" # Question Extracted from the 'Civ-Pro' dataset
options = "" # Options Extracted from the 'Civ-Pro' dataset

input_prompt = f"""
Task:
{system-prompt}

Legal Context:
{legal_context}

Question:
{question}

Options:
{options}

While answering make sure to use the following format:

[Explanation of your legal reasoning step by step as numbered points]

[Final Answer with the correct option]
"""
# The above input-prompt is to be used for the Zero-shot LLM inference

