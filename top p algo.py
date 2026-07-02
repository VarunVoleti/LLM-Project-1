# Use model.generate to use top-p instead of greedy
# You may find this link helpful: https://huggingface.co/docs/transformers/en/main_classes/text_generation
prompt = "Hi my name"

"""
YOUR CODE HERE (~5 lines of code)
"""

input_tokens = tokenizer(prompt, return_tensors="pt").to(model.device)  # tokenizing the prompt and returning the token IDs as a PyTorch tensor

#print(input_tokens)  # printing the token IDs of the prompt

output_tokens = model.generate(
    **input_tokens,  # passing the token IDs of the prompt to the model for text generation
    max_new_tokens=10,  # setting the maximum number of new tokens to generate
    do_sample=True,  # enabling sampling for text generation, greedy is used, which picks highest possibility as the next token
    pad_token_id=tokenizer.eos_token_id,  # setting the padding token ID to the end-of-sequence token ID
    top_p=0.9,  # setting the top-p value for nucleus sampling
    temperature=0.7  # setting the temperature value for sampling
)

output = tokenizer.decode(output_tokens[0], skip_special_tokens=True)  # decoding the output token IDs back into text and skipping special tokens

print(output)


# Write a helper function that wraps model.generate for top-p decoding,
# then test it on multiple prompts
def generate(model, tokenizer, prompt, strategy="top_p", max_new_tokens=128):
    """
    YOUR CODE HERE (~10-15 lines of code)
    """
    pass

    enc = tokenizer(prompt, return_tensors="pt").to(model.device)  # tokenizing the prompt and returning the token IDs as a PyTorch tensor
    gen_args = dict(**enc, max_new_tokens=max_new_tokens,pad_token_id=tokenizer.eos_token_id)    

    if strategy == "greedy":
        gen_args["do_sample"] = False
    elif strategy == "top_p":
        gen_args.update(dict(do_sample=True, top_p=0.9, temperature=0.9))

    output = model.generate(**gen_args)  # generating text using the model with the specified decoding strategy
    return tokenizer.decode(output[0], skip_special_tokens=True)  # decoding the output token IDs back into text and skipping special tokens


tests=["Once upon a time","What is 2+2?", "Suggest a party theme."]
for prompt in tests:
    print(f"\n== GPT-2 | Top-p ==")
    print(generate(model, tokenizer, prompt, "top_p", 32))