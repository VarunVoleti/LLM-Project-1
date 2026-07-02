# %% [markdown]
# # Project 1: Build an LLM Playground
# 
# Welcome to your first project! In this project, you'll build a simple large language model (LLM) playground, an interactive environment where you can experiment with LLMs and understand how they work under the hood.
# 
# The goal here is to understand the foundations and mechanics behind LLMs rather than relying on higher-level abstractions or frameworks. You'll see what happens under the hood, how an LLM receives text, processes it, and generates a response. In later projects, you'll use frameworks like `Ollama` and `LangChain` that simplify many of these steps. But before that, this project will help you build a solid mental model of how LLMs actually work.
# 
# ---
# ## Environment Setup
# We'll use Google Colab, a free browser-based platform that lets you run Python code and machine learning models without installing anything locally. Go to [Google Colab](https://colab.research.google.com/) and upload this notebook to get started.

# %% [markdown]
# If you prefer to run the project locally, you need a reproducible setup. Open a terminal in the same directory as this notebook and run the environment setup commands below to install dependencies and create an isolated environment.
# 
# ### Step 1: Use Conda or uv to install the project dependencies.
# 
# #### Option 1: Conda
# 
# [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html) is an open-source package and environment manager that lets you create isolated environments and install Python and non-Python dependencies together.
# 
# ```bash
# # Create and activate the conda environment
# conda env create -f environment.yaml && conda activate llm_playground
# ```
# 
# #### Option 2: uv (faster)
# 
# [uv](https://docs.astral.sh/uv/) (faster) is a fast Python package installer and virtual environment tool written in Rust that aims to replace pip, pip-tools, and virtualenv with a single, high-performance workflow.
# 
# ```bash
# # Install uv (skip if already installed)
# curl -LsSf https://astral.sh/uv/install.sh | sh
# 
# # Create venv and install dependencies
# uv venv .venv --python 3.11 && source .venv/bin/activate
# uv pip install -r requirements.txt
# ```
# 
# ### Step 2: Register this environment as a Jupyter kernel
# This step is optional. Do it only if your environment doesn’t appear in Jupyter’s kernel list.
# ```bash
# python -m ipykernel install --user --name=llm_playground --display-name "llm_playground"
# ```
# 
# Now switch the kernel to `llm_playground` (Kernel → Change Kernel).

# %% [markdown]
# ---
# ## Learning Objectives  
# - Understand tokenization and how raw text is converted into a sequence of discrete tokens
# - Load and inspect a pretrained LLM (GPT-2) using Hugging Face
# - Understand logits, probabilities, and how models predict the next token
# - Count and explore model parameters to understand model scale
# - Explore decoding strategies: greedy decoding and top-p (nucleus) sampling
# - See the leap from GPT-2 (simple text completion) to a modern model that understands questions and thinks before answering
# - Look ahead: inference engines for serving models in practice
# 
# 
# Let's get started!
# 
# ---

# %%
# Confirm required libraries are installed and working.
import torch, transformers, tiktoken  # tiktoken is a production ready tokenizer for OpenAI models
print("torch", torch.__version__, "| transformers", transformers.__version__)
print("✅ Environment check complete. You're good to go!")

# %% [markdown]
# # 1: Tokenization
# 
# A neural network cannot process raw text directly. It needs numbers.
# Tokenization is the process of converting text into numerical IDs that models can understand. In this section, you will learn how tokenization works in practice and why it is an essential step in every language model pipeline.
# 
# Tokenization methods generally fall into three main categories:
# 1. Word-level
# 2. Character-level
# 3. Subword-level
# 
# ### 1.1 - Word-level tokenization
# This method splits text by whitespace and treats each word as a single token. In the next cell, you will implement a basic word-level tokenizer by building a vocabulary that maps words to IDs and writing `encode` and `decode` functions.

# %%
# Creating a tiny corpus. In practice, a corpus is generally the entire internet-scale dataset used for training and internet data is gathered through "Web crawling".

corpus = [
    "Please do not deploy on friday",
    "it works on my machine",
    "extra spicy noodles are a bad idea!",
    "tokens are tiny pieces of text",
]
# here we are using only 4 short sentences as our entire corpus and we are training word level tokenizer with this corpus

# Step 1: Build vocabulary (all unique words in the corpus) and mappings

# list vs set: list allows duplicates, set does not allow duplicates. We will use a list to store the vocabulary and then convert it to a set to remove duplicates and then convert it back to a list to maintain the order of the words in the corpus

vocab = []   # vocab is a list of all unique words in the corpus
# we will use a simple whitespace tokenizer to split the sentences into words and build the vocabulary and make sure there are no duplicates
word2id = {}
id2word = {}
# word2id and id2word are dictionaries that map words to unique integer ids and vice versa, we can go from any given id to token and token to id aswell
"""
YOUR CODE HERE (~6-15 lines of code)
"""

UNK = "UnKnown"        # can be used to represent any unknown word that is not in the vocabulary,
words = set()          # using set to store unique words in the corpus
for doc in corpus:     # using for loop to iterate through each document in the corpus
    d = doc.lower().split(" ") # splitting the document into words using whitespace as the delimiter using built-in split() method
    for word in d:
        #vocab.append(word) # adding the word to the vocab list
        words.add(word)   # instead of adding words to vocab list, we are adding them to the set to remove duplicates


vocab = [UNK] + sorted(list(words) ) # adding the unique words to the vocab list by converting the set to a list
vocab

#len(vocab)
# gpt 2 vocabulary length is 50257, we are using a very small corpus and we are not using any special tokens like <PAD>, <UNK>, <SOS>, <EOS> etc. so our vocabulary length will be much smaller than 50257

# now we are having perfect vocabulary with no duplicates and we can create the mappings for word2id and id2word, and also completed building a tokenizer for our corpus, we can now use this tokenizer to tokenize any new document and convert it into a list of tokens (words) and then convert those tokens into their corresponding ids using the word2id mapping and also we can convert the ids back to tokens using the id2word mapping

    #print(d)
    #print(doc)

print(vocab)
# encoding and decoding through vocabulary and id's are their indexs 

for index, v in enumerate(vocab):    # In Python, the enumerate() function allows you to iterate over a list (or any sequence) while keeping track of both the index and the value of each item simultaneously
    word2id[v] = index    # mapping each word to its corresponding index in the vocab list
    id2word[index] = v    # mapping each index to its corresponding word in the vocab list
    
    
#  word2id['extra']  # we can use the word2id mapping to get the id of any word in the vocab list, for example, the id of the word 'extra' is 2
#id2word[1]
    #print(index, v)

# even chatgpt works by encoding and decoding, it gets transformed into a sequence of IDs

# %%
# Step 2: Define encode and decode functions
def encode(text):
    # converts text to token IDs
    """
    YOUR CODE HERE (~1-5 lines of code)
    """
    pass
    ids = []
    words = text.lower().split()  # splitting the input text into words using whitespace as the delimiter
    for w in words:
        ids.append(word2id.get(w, word2id[UNK]))   # adding the corresponding id of each word to the ids list using the word2id mapping
    return ids  # returning the list of token IDs

#encode("it works for me")  # testing the encode function with a sample input text

# it only works for the words that are present in the vocab list, if we try to encode a word that is not present in the vocab list, it will throw a KeyError 
# to fix this we can use a special token such as UnKnown


def decode(ids):
    # converts token IDs back to text
    """
    YOUR CODE HERE (~1-5 lines of code)
    """
    pass
    words = []
    for i in ids:
        words.append(id2word[i])   # adding the corresponding word of each id to the words list using the id2word mapping
    return " ".join(words) # joining the list of words into a single string using whitespace as the delimiter
   
   
   # print(encode("it works on my machine"))  # testing the decode function with a sample input text

text = "it works for me" 

ids = encode(text)  # encoding the input text into a list of token IDs
recovered_text = decode(ids)  # decoding the list of token IDs back into text

print(text)  # printing the original input text
print(ids)   # printing the list of token IDs
print(recovered_text)  # printing the recovered text after decoding the token IDs




# %%
# Step 3: Test your tokenizer with some random sentences.
# Try a sentence with unseen words and see what happens (and how to fix it)

"""
YOUR CODE HERE
"""

# %% [markdown]
# While word-level tokenization is simple and easy to understand, it has two key limitations that make it impractical for large-scale models:
# 1. Large vocabulary size: every new word or variation (for example, run, runs, running) increases the total vocabulary, leading to higher memory and training costs.
# 2. Out-of-vocabulary (OOV) problem: the model cannot handle unseen or rare words that were not part of the training vocabulary, so they must be replaced with a generic [UNK] token.
# 
# The next section introduces character-level tokenization, where text is represented as individual characters instead of words.

# %% [markdown]
# ### 1.2 - Character-level tokenization
# 
# In this approach, every single character (including spaces, punctuation, and even emojis) is assigned its own ID.
# 
# In the next cell, we will rebuild a tokenizer using the same corpus as before, but this time with a character-level approach.
# For simplicity, we will only use English letters (a-z, A-Z) and punctuation.

# %%
import string

# Step 1: Create a vocabulary that includes uppercase letters, lowercase letters, and punctuation.
vocab = []
char2id = {}
id2char = {}

"""
YOUR CODE HERE (~5 lines of code)
"""

#string.ascii_lowercase
chars = list(string.ascii_lowercase + string.ascii_lowercase + string.punctuation)
vocab = ["[UnKnown]"] + chars  # adding a special token for unknown characters to the vocabulary

#print (vocab)  # printing the vocabulary 

char2id = {ch: idx for idx, ch in enumerate(vocab)}  # creating a mapping from characters to their corresponding indices in the vocabulary
id2char = {idx: ch for ch, idx in char2id.items()}  # creating a mapping from indices to their corresponding characters in the vocabulary

print(char2id)  # printing the character to index mapping

print(id2char)  # printing the index to character mapping

# %%
# Step 2: Implement encode() and decode() functions to convert between text and IDs.
def encode(text):
    # convert text to list of IDs
    """
    YOUR CODE HERE (~2-5 lines of code)
    """
    pass

    ids = []
    for c in text:
        ids.append(char2id.get(c, char2id["[UnKnown]"]))
    
    return ids  # returning the list of character IDs



def decode(ids):
    # Convert list of IDs to text
    """
    YOUR CODE HERE (~2-5 lines of code)
    """
    pass
    char = []
    for i in ids:
        char.append(id2char[i])
    return "".join(char)  # joining the list of characters into a single string using whitespace as the delimiter



ids = encode("Hi howdy?")  # testing the encode function with a sample input text
recovered_text = decode(ids)  # decoding the list of character IDs back into text

print(ids)  # printing the list of character IDs
print(recovered_text)  # printing the recovered text after decoding the character IDs

# %%
# Step 3: Test your tokenizer on a some sample words.
"""
YOUR CODE HERE (~2-5 lines of code)
"""

# %% [markdown]
# Character-level tokenization solves the out-of-vocabulary problem but introduces new challenges:
# 
# 1. Longer sequences: because each word becomes many tokens, models need to process much longer inputs.
# 2. Weaker semantic representation: individual characters carry very little meaning, so models must learn relationships across many steps.
# 3. Higher computational cost: longer sequences lead to more tokens per input, which increases training and inference time.
# 
# To find a better balance between vocabulary size and sequence length, we move to subword-level tokenization next.

# %% [markdown]
# ### 1.3 - Subword-level tokenization
# 
# Subword methods such as `Byte-Pair Encoding (BPE)`, `WordPiece`, and `SentencePiece` **learn** common groups of characters and merge them into tokens. For example, the word **unbelievable** might turn into three tokens: **["un", "believ", "able"]**. This approach strikes a balance between word-level and character-level methods and fixes their limitations.
# 
# The BPE algorithm builds a vocabulary iteratively using the following process:
# 1. Start with individual characters or bytes (each character is a token).
# 2. Count all adjacent pairs of tokens in a large text corpus.
# 3. Merge the most frequent pair into new tokens.
# 
# Repeat steps 2 and 3 until you reach the desired vocabulary size (for example, 50,000 tokens).
# 
# In the next cell, you will experiment with BPE in practice to see how it compresses text into meaningful subword units. Instead of implementing the algorithm from scratch, you will use a pretrained tokenizer, which was already trained on a large text corpus to build its vocabulary, such as the data used to train `GPT-2`. This allows you to see how BPE works in practice with a real, learned vocabulary.

# %%
from transformers import AutoTokenizer

# Step 1: Load a pretrained GPT-2 tokenizer from Hugging Face. 
# Refer to this to learn more: https://huggingface.co/docs/transformers/en/model_doc/gpt2

"""
YOUR CODE HERE (~1 line of code)
"""

bpe_tok = AutoTokenizer.from_pretrained("gpt2")  # loading the pretrained GPT-2 tokenizer from Hugging Face
bpe_tok 


# %%
#bpe_tok.vocab_size  # printing the size of the vocabulary of the GPT-2 tokenizer
bpe_tok.all_special_tokens  # printing all the special tokens used by the GPT-2 tokenizer

# %%
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

# %% [markdown]
# ### 1.4 - TikToken
# 
# `tiktoken` is a fast, production-ready library for tokenization used by OpenAI models.
# It is designed for efficiency and consistency with how OpenAI counts tokens in GPT models.
# 
# In this section, you will explore how different model families use different tokenizers. We will compare tokenizers used to train `GPT-2` and more powerful models such as `GPT-4`. By trying both, you will see how tokenization has evolved to handle more diverse text (including emojis and special characters) while remaining efficient.
# 
# In the next cell, you will use tiktoken to load these encodings and inspect how each one splits the same text. You may find reading this doc helpful: https://github.com/openai/tiktoken

# %%
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

# %% [markdown]
# Try changing the input sentence and observe how different tokenizers behave.
# Experiment with:
# - Emojis, special characters, or punctuation
# - Code snippets or structured text
# - Non-English text (for example, Japanese, French, or Arabic)
# 
# If you are curious, you can also attempt to implement the BPE algorithm yourself using a small text corpus to see how token merges are learned in practice.
# 
# ### 1.5 - Key Takeaways
# - **Word-level**: simple and intuitive, but limited by large vocabularies and out-of-vocabulary issues
# - **Character-level**: flexible and covers all text, but produces long sequences that are harder to model
# - **Subword / BPE**: balances both worlds and is the default choice for most modern LLMs
# - **TikToken**: a production-ready tokenizer used in OpenAI models, demonstrating how optimized subword vocabularies are applied in real systems

# %% [markdown]
# # 2: What is a Language Model?
# 
# At its core, a **language model** is a function that predicts the next token. Given a sequence of tokens `[t₁, t₂, …, tₙ]`, it outputs probabilities for the next token `tₙ₊₁`. 
# 
# Models like GPT-2 use many stacked Transformer layers. Each layer mixes information between tokens (attention) and transforms it (feed-forward). Together, these layers learn patterns in text. At the end, the model produces logits: one score per token in the vocabulary. Higher logits mean the token is more likely to be next.
# 
# In this section, you’ll load GPT-2, look at its architecture, count its parameters, and see how it predicts the next token.

# %% [markdown]
# ### 2.1: Loading GPT-2
# 
# There are different ways to load and run pretrained language models.
# 
# In this project, we’ll use Hugging Face Transformers, a popular Python library that makes it easy to download models like GPT-2 and run them locally.
# 
# There are also dedicated inference engines for serving and running modern LLMs more efficiently, such as **Ollama**, **SGLang**, and **vLLM**. We’ll explore those in future projects.
# 
# In the next cell, you’ll load GPT-2 and inspect its architecture.

# %%
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


# %% [markdown]
# ### 2.2: Counting Parameters
# 
# You often hear that an LLM has a few million or billion parameters. But what does that actually mean? Every weight and bias value inside every layer is a parameter. These are the numbers that the model learned during training.
# 
# Next, you will count the total number of parameters in GPT-2 and break them down by component to see where most of the model's capacity lives.

# %%
# Count the total number of parameters in GPT-2
# Hint: use model.parameters() and the .numel() method on each parameter tensor
"""
YOUR CODE HERE (~2-3 lines of code)
"""

total = sum(p.numel() for p in gpt2.parameters())
print(f"Total number of parameters in GPT-2: {total:,}")  # printing the total number of parameters in the GPT-2 model

# %% [markdown]
# **Think about scale:** GPT-2 Small has 124M parameters. GPT-4 is estimated to have over 1 trillion. If each parameter is a 16-bit floating point number (2 bytes), how much memory would you need just to store GPT-2 in RAM? What about a 70B parameter model?

# %% [markdown]
# ### 2.3: From Text to Predictions
# 
# When you pass a sequence of tokens through a language model, it produces a tensor of logits with shape
# `(batch_size, seq_len, vocab_size)`.
# Each position in the sequence receives a vector of scores representing how likely every possible token is to appear next. By applying a softmax function on the last dimension, these logits can be converted into probabilities that sum to 1.
# 
# In the next cell, you will feed a sentence into GPT-2, print the shape of its logits, and display the five most likely next tokens predicted for the final position in the sequence.

# %%
from transformers import AutoTokenizer

# Step 1: Load the GPT-2 tokenizer (the model is already loaded above)
"""
YOUR CODE HERE (~1 line of code)
"""

tokenizer = AutoTokenizer.from_pretrained("gpt2")  # loading the GPT-2 tokenizer using the Hugging Face transformers library
tokenizer  # printing the tokenizer object to inspect its properties

# %%
# Step 2: Tokenize input text
text = "Hello my name"

"""
YOUR CODE HERE (~1 line of code)
"""

input_ids = tokenizer(text, return_tensor="pt").input_ids  # tokenizing the input text and returning the token IDs as a PyTorch tensor

print(input_ids)  # printing the token IDs of the input text

# %%
# Step 3: Pass the input IDs to the model
""" YOUR CODE HERE (~2-3 lines of code) """
with torch.no_grad():
    # Convert list to a PyTorch tensor and add batch dimension
    tensor_input = torch.tensor([input_ids])
    outputs = gpt2(tensor_input)
    logits = outputs.logits

print(logits.shape) # printing the shape of the output logits tensor


# %%
import torch.nn.functional as F

# Step 4: Predict the next token
# Take the logits from the final position, apply softmax to get probabilities,
# and then extract the top 5 most likely next tokens. 
# You may find F.softmax and torch.topk helpful in your implementation.

"""
YOUR CODE HERE (~3-7 lines of code)
"""

probs = F.softmax(logits[0, -1], dim=-1)  # applying softmax to the logits of the final position to get probabilities

topk = torch.topk(probs, 5)  # extracting the top 5 most likely next tokens using torch.topk

print("\n Top 5 next token predictions:")
for idx, p in zip(topk.indices.tolist(), topk.values.tolist()):
    print(f"Token: {tokenizer.decode([idx]):>10s} - {p:.4f}")  # printing the top 5 next token predictions along with their probabilities

# %% [markdown]
# ### 2.4: Key Takeaway
# 
# A language model is not a black box or something mysterious.
# It is a large composition of simple, understandable layers such as attention and feed-forward networks, trained together to predict the next token in a sequence.
# 
# By learning this next-token prediction task at scale, the model gradually develops an internal understanding of language structure, meaning, and context, which allows it to generate coherent and relevant text.

# %% [markdown]
# # 3: Text Generation (Decoding)
# Once a language model can predict the next-token probabilities, we can use it to generate text. This is called text generation or decoding.
# 
# Conceptually, generation is a loop:
# 
# 1. Feed the current token sequence into the model.
# 
# 2. The model outputs a probability distribution over the next token.
# 
# 3. A decoding algorithm picks the next token from that distribution.
# 
# 4. Append the chosen token to the sequence and repeat.
# 
# In practice, libraries like Hugging Face Transformers provide generate() methods that handle this loop for you, including stopping conditions, batching, and efficiency tricks.
# 
# Different decoding strategies control how the next token is chosen, and how creative or deterministic the output feels:
# 
# - **Greedy decoding**: always pick the token with the highest probability. Simple and consistent, but often repetitive.
# 
# - **Top-p (nucleus) sampling**: randomly sample from the smallest set of tokens whose cumulative probability exceeds a threshold p. This adds variety while keeping outputs coherent.
# 
# In the following sections, you'll use generate() to produce text from GPT-2 using both greedy decoding and top-p sampling.

# %% [markdown]
# ### 3.1: Greedy Decoding
# In this section, you will use GPT-2 and Hugging Face's built-in generate method to produce text using the greedy decoding strategy.

# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "gpt2"

device = "cuda" if torch.cuda.is_available() else "mps"  # checking if a GPU is available and setting the device accordingly

# Step 1. Load GPT-2 model and tokenizer.
"""
YOUR CODE HERE (~2 lines of code)
"""

tokenizer = AutoTokenizer.from_pretrained(model_id)  # loading the GPT-2 tokenizer using the Hugging Face transformers library
model = AutoModelForCausalLM.from_pretrained(model_id).eval().to(device)  # loading the GPT-2 model using the Hugging Face transformers library

# Step 2. tokenize the prompt, then use model.generate() to generate text, then print the text
prompt = "Hi my name"

"""
YOUR CODE HERE (~3-6 lines of code)
"""

input_tokens = tokenizer(prompt, return_tensors="pt").to(model.device)  # tokenizing the prompt and returning the token IDs as a PyTorch tensor

#print(input_tokens)  # printing the token IDs of the prompt

output_tokens = model.generate(
    **input_tokens,  # passing the token IDs of the prompt to the model for text generation
    max_new_tokens=10,  # setting the maximum number of new tokens to generate
    do_sample=False  # enabling sampling for text generation, greedy is used, which picks highest possibility as the next token

)

# %%
output_tokens  # printing the output token IDs generated by the model

# %%
# Step 3: Write a helper function that wraps model.generate for greedy decoding,
# then test it on multiple prompts to observe repetition patterns.

tests = ["Once upon a time", "What is 2+2?", "Suggest a party theme."]

def generate(model, tokenizer, prompt, max_new_tokens=32):
    """
    YOUR CODE HERE (~5 lines of code)
    """
    pass

    enc = tokenizer(prompt, return_tensors="pt").to(model.device)  # tokenizing the prompt and returning the token IDs as a PyTorch tensor
    output = model.generate(
        **enc,
         max_new_tokens=max_new_tokens, 
         do_sample=False,
         pad_token_id=tokenizer.eos_token_id
    )  # generating text using the model with greedy decoding
    return tokenizer.decode(output[0], skip_special_tokens=True)  # decoding the output token IDs back into text and skipping special tokens

for prompt in tests:
    print(f"\n GPT-2 | Greedy")
    print(generate(model, tokenizer, prompt, 80))

# %% [markdown]
# Naively selecting the single most probable token at each step (known as greedy decoding) often leads to poor results in practice:
# - Repetition loops: phrases like "The cat is is is…"
# - Short-sighted choices: the most likely token right now might lead to incoherent text later
# 
# These issues are why more advanced decoding methods such as top-p (nucleus) sampling are commonly used to make model outputs more diverse and natural.

# %% [markdown]
# ### 3.2: Top-p (Nucleus) Sampling
# Top-p sampling (also called nucleus sampling) introduces controlled randomness into text generation. Instead of always picking the single most likely token, it samples from the smallest set of tokens whose cumulative probability exceeds a threshold `p` (e.g., 0.9).
# 
# This allows the model to explore multiple plausible continuations, producing more diverse and natural-sounding text while still staying coherent.
# 
# In this section, you will implement a generate function that supports both greedy and top-p strategies.

# %%
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




# %%
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

# %% [markdown]
# # 4: From Simple GPT2 to Modern LLMs
# 
# So far, we have used **GPT-2**, one of the earliest publicly available language models (released in 2019, 124M parameters). GPT-2 can only do one thing: **complete text**. Given some input, it predicts what words come next. It has no concept of questions, instructions, or conversation. If you type `"What is 2+2?"`, GPT-2 will continue the text as if it were part of a web page or article. It does not understand you are asking a question.
# 
# Modern language models are a different. Models like **Qwen3-0.6B** (2025, 600M parameters) have gone through additional training stages that unlock fundamentally new capabilities:
# 
# - **Instruction following**: they interpret your input as a request and produce a helpful response
# - **Conversation**: they maintain context across a multi-turn dialogue
# - **Reasoning**: they can *think step-by-step* before answering, using internal reasoning tokens (`<think>...</think>`) to work through a problem before giving a final answer
# 
# In this section, you will see this dramatic contrast firsthand by giving the same prompts to both models. In week 4, we'll learn about thinking and reasoning models in detail.
# 
# ### 4.1: Chat Templates
# 
# Instruction-tuned models expect input in a structured **chat format**, not raw text. Instead of receiving `"What is 2+2?"` as plain text, the model expects a formatted message like:
# 
# ```
# <|user|>What is 2+2?<|assistant|>
# ```
# 
# Each model family defines its own format. The Hugging Face `tokenizer.apply_chat_template()` method handles this formatting automatically. Without it, even an instruction-tuned model receives unstructured text and falls back to simple completion behavior.
# 
# ### 4.2: GPT-2 vs. Qwen3-0.6B
# 
# In the next cells, you will load both models and feed the same prompt to each one:
# 
# - **GPT-2**: receives the raw prompt and blindly continues the text
# - **Qwen3-0.6B**: receives a properly formatted chat message, *thinks* through the problem, and produces a direct answer

# %%
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

# %% [markdown]
# Both models are now loaded. GPT-2 has 124M parameters; Qwen3-0.6B has roughly 600M. If the previous cell took some time, that was mainly due to model download. The models are cached locally, so future runs will be much faster.
# 
# Next, we will generate text from both models using the same prompt. For GPT-2, we pass the raw text directly. For Qwen, we first format the prompt as a chat message using `apply_chat_template()`, then generate. Both use **top-p sampling** so the outputs are varied and natural.

# %%

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







# %% [markdown]
# # 5. (Optional) A Small Interactive LLM Playground
# This section is optional. You do not need to implement it to complete the project. It is meant purely for exploration and will not significantly affect your core AI engineering skills.
# 
# If you are curious, you can build a simple interactive playground to experiment with text generation. You can:
# - Create input widgets for the prompt, model selection, decoding strategy, and temperature
# - Use Hugging Face's generate method to produce text based on the selected settings
# - Display the model's response directly in the notebook output
# 
# You may find following links helpful:
# - https://ipywidgets.readthedocs.io/en/latest/
# - https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html

# %%
import ipywidgets as widgets
from IPython.display import display, Markdown
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Steps to implement: 
# 1. Load models and tokenizers (GPT-2 and Qwen).
# 2. Define a helper function to generate text with different decoding strategies.
# 3. Create interactive UI elements (prompt box, model selector, strategy selector, temperature slider).
# 4. Add a button to trigger text generation.
# 5. Define the button’s behavior.
# 6. Display the full UI for the playground.

try:
    tokenizers
    models
except NameError:
    # Use a verified open-source identifier if Qwen3-0.6B isn't local
    MODELS = {
        "gpt2": "gpt2",
        "qwen": "Qwen/Qwen1.5-0.5B"  
    }
    tokenizers, models = {}, {}
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    for key, mid in MODELS.items():
        tok = AutoTokenizer.from_pretrained(mid)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        mdl = AutoModelForCausalLM.from_pretrained(mid).eval().to(device)
        tokenizers[key] = tok
        models[key] = mdl
        print(f"Loaded {mid} as {key}")

def generate_text(model_key, prompt, strategy="greedy", temperature=1.0, max_new_tokens=32):
    tok, mdl = tokenizers[model_key], models[model_key]
    
    # Apply chat template for instruction-tuned models like Qwen
    if "qwen" in model_key.lower():
        messages = [{"role": "user", "content": prompt}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) #

    enc = tok(prompt, return_tensors="pt").to(mdl.device)
    gen_args = dict(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)
    
    if strategy == "greedy":
        gen_args["do_sample"] = False
    elif strategy == "top_k":
        gen_args.update(dict(do_sample=True, top_k=50, temperature=temperature))
    elif strategy == "top_p":
        gen_args.update(dict(do_sample=True, top_p=0.9, temperature=temperature))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
        
    out = mdl.generate(**gen_args)
    
    # Strip prompt tokens out of output for instruction tuning clarity
    new_tokens = out[0][enc["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True)

# --- 3. Create interactive UI elements ---
prompt_box = widgets.Textarea(
    value="Once upon a time",
    placeholder="Type your prompt here...",
    description="Prompt:",
    layout=widgets.Layout(width='100%', height='100px')
)

model_dropdown = widgets.Dropdown(
    options=[("GPT-2", "gpt2"), ("Qwen", "qwen")],
    value="gpt2",
    description="Model:"
)

strategy_dropdown = widgets.Dropdown(
    options=[("Greedy", "greedy"), ("Top-K", "top_k"), ("Top-P", "top_p")],
    value="greedy",
    description="Strategy:"
)

temperature_slider = widgets.FloatSlider(
    value=1.0,
    min=0.1,
    max=2.0,
    step=0.1,
    description="Temperature:"
)

# --- 4. Add a button to trigger text generation ---
generate_button = widgets.Button(description="Generate Text", button_style='success')
output_area = widgets.Output()

# --- 5. Define the button's behavior ---
def on_generate(_):
    output_area.clear_output()
    with output_area:
        # Gray out button to show it's working
        generate_button.disabled = True
        generate_button.description = "Generating..."
        try:
            # Fixed function call name match
            result = generate_text(
                model_dropdown.value,
                prompt_box.value,
                strategy_dropdown.value,
                temperature_slider.value
            )
            display(Markdown(f"**Generated Text:**\n\n{result}"))
        except Exception as e:
            display(Markdown(f"**Error:** {e}"))
        finally:
            generate_button.disabled = False
            generate_button.description = "Generate Text"

# Link button click event to handler
generate_button.on_click(on_generate)

# --- 6. Display the full UI for the playground ---
ui_layout = widgets.VBox([
    prompt_box,
    widgets.HBox([model_dropdown, strategy_dropdown, temperature_slider]),
    generate_button,
    output_area
])

display(ui_layout)


# %% [markdown]
# # 6: Inference Engines: Ollama, vLLM, SGLang
# 
# So far, we loaded models directly in Python using HuggingFace's `transformers` library. This is great for learning, but in practice models run as **servers** that expose an API. Client applications send requests and receive responses over HTTP — the model itself stays loaded in memory (and on the GPU) between requests.
# 
# An **inference engine** handles all the heavy lifting: model loading, GPU memory management, request batching, and serving an HTTP API. Popular inference engines include:
# 
# | Engine | Best for |
# |--------|----------|
# | **Ollama** | Easy local use and experimentation |
# | **vLLM** | High-throughput production serving |
# | **SGLang** | Fast serving + structured output |
# 
# Most inference engines expose an **OpenAI-compatible API**. This means you can learn one client library (the `openai` Python package) and swap backends freely: Ollama for local development, vLLM or SGLang for production.
# 
# In future weeks, we'll learn about Ollama, set it up, and use it to easily load and build on top of modern powerful LLMs!

# %% [markdown]
# ## 🎉 Congratulations!
# 
# You've just explored the internals of a real **LLM**. In this project you:
# * Learned how **tokenization** works — from word-level to BPE — and why it matters
# * Used `tiktoken` to compare tokenizers across different model generations
# * Loaded GPT-2 and inspected its Transformer blocks and layers
# * **Counted parameters** and understood where a model's capacity lives
# * Learned how the model produces **logits and probabilities** to predict the next token
# * Explored **decoding strategies**: greedy decoding and top-p (nucleus) sampling
# * Witnessed the leap from GPT-2 (simple text completion) to Qwen3-0.6B — a modern model that **understands questions and thinks before answering**
# * Learned about **inference engines** (Ollama, vLLM, SGLang) and the OpenAI-compatible API pattern
# 
# 👏 **Great job!** Take a moment to celebrate. You now have a working mental model of how LLMs work — from raw text input all the way to generated output. The skills and intuitions you built here will serve as the foundation for everything that comes next.
# 
# 


