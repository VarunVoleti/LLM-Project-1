#bpe_tok.vocab_size  # printing the size of the vocabulary of the GPT-2 tokenizer
bpe_tok.all_special_tokens  # printing all the special tokens used by the GPT-2 tokenizer

# Step 2: Encode a sample sentence and inspect how BPE breaks it into subword tokens.
# Hint: bpe_tok has encode, decode, and convert_ids_to_tokens methods.

"""
YOUR CODE HERE (~5-8 lines of code)
"""

sample = "Unbelievable tokenization!"  

#bpe_tok.encode(sample)  # encoding the sample sentence into a list of token IDs using the GPT-2 tokenizer

ids = bpe_tok.encode(sample)  # encoding the sample sentence into a list of token IDs using the GPT-2 tokenizer
recover = bpe_tok.decode(ids)  # decoding the list of token IDs back into text using the GPT-2 tokenizer

#print(ids)  # printing the list of token IDs
#print(recover)  # printing the recovered text after decoding the token IDs

print("\nInput text :", sample)  # printing the original input text
print("Token IDs  :", ids)  # printing the list of token IDs
print("Tokens :", bpe_tok.convert_ids_to_tokens(ids))  # printing the list of tokens corresponding to the token IDs
print("Decoded :", recover)  # printing the recovered text after decoding the token IDs