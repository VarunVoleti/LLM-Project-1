import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load both GPT-2 and Qwen/Qwen3-0.6B models using HuggingFace `.from_pretrained` method.
"""
YOUR CODE HERE (~5-15 lines of code)
"""

MODELS = {
    "GPT-2": "gpt2",
    "qwen": "Qwen/Qwen3-0.6B"}

tokenizers, models = {}, {}
device = "cuda" if torch.cuda.is_available() else "mps"  # checking if a GPU is available and setting the device accordingly

for key, mid in MODELS.items():
    tok = AutoTokenizer.from_pretrained(mid)  # loading the tokenizer for the model using the Hugging Face transformers library
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token  # setting the padding token to the end-of-sequence token if the padding token is not defined
    
    mdl = AutoModelForCausalLM.from_pretrained(mid).eval().to(device)  # loading the model using the Hugging Face transformers library
    tokenizers[key] = tok, models[key] = tok, mdl  # storing the tokenizer and model in the respective dictionaries
    print(f"Loaded {mid} as {key}")  #  printing a message indicating that the model has been loaded successfully


# Generate text from both models using the same prompt.
# For GPT-2, pass the raw prompt directly.
# For Qwen, use apply_chat_template() to format it as a chat message before generating.
# Refer to: https://huggingface.co/docs/transformers/en/chat_templating
# Use top-p (nucleus) sampling for both models.

#prompt = "What is 2+2?"

# --- GPT-2: simple text completion ---

"""
YOUR CODE HERE (~5 lines of code)
"""
"""
gpt2_inputs = tokenizers["GPT-2"](prompt, return_tensors="pt").to(device)  # tokenizing the prompt for GPT-2 and returning the token IDs as a PyTorch tensor
gpt2_output = models["GPT-2"].generate(
    **gpt2_inputs,  # passing the token IDs of the prompt to the GPT-2 model for text generation
    max_new_tokens=32,  # setting the maximum number of new tokens to generate
    do_sample=True,  # enabling sampling for text generation
    top_p=0.9,  # setting the top-p value for nucleus sampling
    temperature=0.7,  # setting the temperature value for sampling
    pad_token_id=tokenizers["GPT-2"].pad_token_id  # setting the padding token ID to the padding token ID of the GPT-2 tokenizer
)

print("\n=== GPT-2 Output ===")
print(tokenizers["GPT-2"].decode(gpt2_output[0], skip_special_tokens=True))  # decoding the output token IDs back into text and skipping special tokens

# --- Qwen3-0.6B: instruction-tuned with chat template ---
"""
"""
YOUR CODE HERE (~5 lines of code)
"""

prompt = "What is 2+2?"

# Automatically locate the correct keys from your dictionary
gpt2_key = next((k for k in tokenizers.keys() if "gpt" in k.lower()), None)
qwen_key = next((k for k in tokenizers.keys() if "qwen" in k.lower()), None)

# Print a helpful diagnostic if the models aren't loaded in the dictionary
if not gpt2_key or not qwen_key:
    print(f"Missing keys! Current dictionary keys are: {list(tokenizers.keys())}")
    print("Please re-run your model initialization cell above first.")
else:
    # --- GPT-2: simple text completion ---
    gpt2_tokenizer = tokenizers[gpt2_key]
    if isinstance(gpt2_tokenizer, tuple):
        gpt2_tokenizer = gpt2_tokenizer[0]
        
    gpt2_inputs = gpt2_tokenizer(prompt, return_tensors="pt").to(device)
    gpt2_output = models[gpt2_key].generate(
        **gpt2_inputs,
        max_new_tokens=32,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        pad_token_id=gpt2_tokenizer.pad_token_id
    )
    print("\n=== GPT-2 Output ===")
    print(gpt2_tokenizer.decode(gpt2_output[0], skip_special_tokens=True))

    # --- Qwen: instruction-tuned with chat template ---
    qwen_tokenizer = tokenizers[qwen_key]
    if isinstance(qwen_tokenizer, tuple):
        qwen_tokenizer = qwen_tokenizer[0]
        
    messages = [{"role": "user", "content": prompt}]
    
    chat_text = qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    qwen_inputs = qwen_tokenizer(chat_text, return_tensors="pt").to(device)
    
    qwen_output = models[qwen_key].generate(
        **qwen_inputs,
        max_new_tokens=32,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        pad_token_id=qwen_tokenizer.eos_token_id
    )
    
    qwen_new_token = qwen_output[0][qwen_inputs["input_ids"].shape[1]:]
    print("\n=== Qwen Output ===")
    print(qwen_tokenizer.decode(qwen_new_token, skip_special_tokens=True))





