import torch
from transformers import GPT2LMHeadModel

# Step 1: Load the smallest GPT-2 model (124M parameters) using the Hugging Face transformers library.
# Refer to: https://huggingface.co/docs/transformers/en/model_doc/gpt2
"""
YOUR CODE HERE (~1 line of code)
"""

gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")  # loading the smallest GPT-2 model (124M parameters) using the Hugging Face transformers library

# Step 2: Print the full model architecture
"""
YOUR CODE HERE (~1 line of code)
"""

#print(gpt2)  # printing the model architecture of the GPT-2 model


# Step 3: Inspect the first Transformer block by printing its layers.
"""
YOUR CODE HERE (~1 line of code)
"""

block = gpt2.transformer.h[0]  # accessing the first Transformer block of the GPT-2 model
print(block)  # printing the layers of the first Transformer block of the GPT-2 model
