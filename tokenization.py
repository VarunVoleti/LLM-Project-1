from transformers import AutoTokenizer

# Step 1: Load the GPT-2 tokenizer (the model is already loaded above)
"""
YOUR CODE HERE (~1 line of code)
"""

tokenizer = AutoTokenizer.from_pretrained("gpt2")  # loading the GPT-2 tokenizer using the Hugging Face transformers library
tokenizer  # printing the tokenizer object to inspect its properties

# Step 2: Tokenize input text
text = "Hello my name"

"""
YOUR CODE HERE (~1 line of code)
"""

input_ids = tokenizer(text, return_tensor="pt").input_ids  # tokenizing the input text and returning the token IDs as a PyTorch tensor

print(input_ids)  # printing the token IDs of the input text

# Step 3: Pass the input IDs to the model
""" YOUR CODE HERE (~2-3 lines of code) """
with torch.no_grad():
    # Convert list to a PyTorch tensor and add batch dimension
    tensor_input = torch.tensor([input_ids])
    outputs = gpt2(tensor_input)
    logits = outputs.logits

print(logits.shape) # printing the shape of the output logits tensor
