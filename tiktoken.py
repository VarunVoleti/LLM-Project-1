import tiktoken

# Compare GPT-2 and GPT-4 tokenizers using tiktoken.

# Step 1: Load two tokenizers
"""
YOUR CODE HERE (~2-3 lines of code)
"""

encodings =[ 
    ("GPT-2", tiktoken.get_encoding("gpt2")),  # loading the GPT-2 tokenizer using tiktoken
    ("GPT-4", tiktoken.get_encoding("cl100k_base"))  # loading the GPT-4 tokenizer using tiktoken
]

# Step 2: Encode the same sentence with both and observe how they differ
sentence = "The 🌟 star-programmer implemented AGI overnight."

"""
YOUR CODE HERE (~3-10 lines of code)
"""

for name, enc in encodings:
    print(f"\n==={name} ===")
    print("Vocab size:", enc.n_vocab)  # printing the vocabulary size of the tokenizer

    # Encode the sentence and print the token IDs
    ids = enc.encode(sentence)  # encoding the sample sentence into a list of token IDs using the tokenizer
    tokens = [enc.decode([i]) for i in ids]  # decoding each token ID back into text using the tokenizer
    print(f"Sentence split into {len(ids)} tokens:", ids)  # printing the list of token IDs
    print(list(zip(ids, tokens)))  # printing the list of token IDs and their corresponding tokens

# Step 3: Inspect special tokens used in each tokenizer

"""
YOUR CODE HERE (~3-10 lines of code)
"""