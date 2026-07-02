# Project 1: Build an LLM Playground — Notes

Understanding how large language models work under the hood: tokenization, model internals, next-token prediction, and decoding strategies — built with GPT-2 and Qwen3-0.6B using Hugging Face transformers and tiktoken.

## Overview

This project skips high-level frameworks (like Ollama or LangChain, used in later projects) to build intuition for what actually happens between raw text input and generated text output. It's organized into 6 parts:


### Tokenization (word → character → subword/BPE → tiktoken)
What is a language model (loading GPT-2, counting parameters, logits → probabilities)
Text generation / decoding (greedy vs. top-p sampling)
From simple GPT-2 to modern instruction-tuned LLMs (Qwen3-0.6B, chat templates)
(Optional) Interactive playground UI with ipywidgets
Inference engines overview (Ollama, vLLM, SGLang)


### Environment Setup

Two options for a reproducible local environment:

**Conda**

bashconda env create -f environment.yaml && conda activate llm_playground

**uv (faster)**

Register as a Jupyter kernel if needed:
```
# Confirm required libraries are installed and working.
import torch, transformers, tiktoken  # tiktoken is a production ready tokenizer for OpenAI models
print("torch", torch.__version__, "| transformers", transformers.__version__)
print("✅ Environment check complete. You're good to go!")
```

Core libraries: torch, transformers, tiktoken, ipywidgets.


## 1. Tokenization

Neural networks can't process raw text — everything has to become numbers first. Tokenization converts text into a sequence of integer IDs. Three approaches were implemented/explored, in increasing sophistication:

### 1.1 Word-level tokenization


Build a vocabulary of unique words from a small corpus, mapped via word2id / id2word dictionaries.
encode() lowercases and splits text on whitespace, looking up each word's ID (falling back to an UnKnown token for out-of-vocabulary words).
decode() reverses the mapping and joins words back with spaces.
Limitations:

Vocabulary explodes with every new word/inflection (run, runs, running → 3 separate entries).
Out-of-vocabulary (OOV) words must be mapped to a generic [UNK] token, losing information.
```
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
```




### 1.2 Character-level tokenization


Vocabulary = every individual character (letters + punctuation), each with its own ID.
Same encode/decode pattern, but operating character-by-character instead of word-by-word.
Trade-offs:

Solves the OOV problem (any text can be represented).
Produces much longer sequences (more tokens per input → higher compute cost).
Individual characters carry very little semantic meaning on their own, so the model has to work harder to learn relationships across many more steps.
```
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
```
```
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
```
```
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
```



### 1.3 Subword-level tokenization (BPE)


Byte-Pair Encoding (BPE), along with WordPiece and SentencePiece, is the standard approach in modern LLMs. It learns a vocabulary from data rather than using fixed rules:

Start with individual characters/bytes as tokens.
Count all adjacent token pairs across a large corpus.
Merge the most frequent pair into a new token.
Repeat until a target vocabulary size is reached (e.g., 50,000).

```
from transformers import AutoTokenizer

# Step 1: Load a pretrained GPT-2 tokenizer from Hugging Face. 
# Refer to this to learn more: https://huggingface.co/docs/transformers/en/model_doc/gpt2

"""
YOUR CODE HERE (~1 line of code)
"""

bpe_tok = AutoTokenizer.from_pretrained("gpt2")  # loading the pretrained GPT-2 tokenizer from Hugging Face
bpe_tok
#bpe_tok.vocab_size  # printing the size of the vocabulary of the GPT-2 tokenizer
bpe_tok.all_special_tokens  # printing all the special tokens used by the GPT-2 tokenizer
```

```
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
```



Example: "unbelievable" → ["un", "believ", "able"].
Used the pretrained GPT-2 tokenizer (AutoTokenizer.from_pretrained("gpt2")) via Hugging Face to see BPE applied in practice, using .encode(), .decode(), and .convert_ids_to_tokens().


### 1.4 tiktoken


tiktoken is OpenAI's fast, production-grade tokenizer, used to keep token counting consistent with how OpenAI models actually process text.
Compared GPT-2's tokenizer (tiktoken.get_encoding("gpt2")) against GPT-4's (cl100k_base) on the same sentence (including an emoji and a hyphenated compound word) to see how vocabulary design has evolved — GPT-4's tokenizer handles emojis/special characters more efficiently and produces different segmentations for the same input.
```
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
```

### 1.5 Key takeaways

MethodProsConsWord-levelSimple, intuitiveHuge vocab, OOV problemCharacter-levelHandles any textLong sequences, weak semantics per tokenSubword / BPEBalances vocab size & sequence lengthSlightly more complex to implementtiktokenProduction-grade, OpenAI-consistentN/A — used for inspection/comparison

BPE (or similar subword methods) is the default choice for essentially all modern LLMs.


## 2. What Is a Language Model?

At its core: a function that predicts the next token given a sequence of previous tokens. GPT-2 stacks multiple Transformer blocks, each combining:


Attention — mixes information between tokens in the sequence.
Feed-forward layers — transforms that mixed representation.


The final output is a set of logits — one raw score per vocabulary token — indicating how likely each token is to come next.

### 2.1 Loading GPT-2
```
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

total = sum(p.numel() for p in gpt2.parameters())
print(f"Total number of parameters in GPT-2: {total:,}")  # printing the total number of parameters in the GPT-2 model
```

Inspecting gpt2.transformer.h[0] shows the internal structure of a single Transformer block (attention + MLP layers).

### 2.2 Counting parameters

GPT-2 Small has 124M parameters; GPT-4 is estimated at over 1 trillion. A useful mental exercise: at 16-bit precision (2 bytes/parameter), 124M params ≈ 248 MB just to store weights — scaling this to a 70B-parameter model implies ~140 GB, illustrating why large models require serious infrastructure (multi-GPU setups, quantization, etc.) just to load into memory.

```
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
```

### 2.3 From text to predictions (logits → probabilities)

Logits have shape (batch_size, seq_len, vocab_size) — one score vector per position, per possible vocabulary token.
Softmax converts the final position's logits into a probability distribution over the whole vocabulary.
torch.topk extracts the 5 most probable next tokens for inspection.
```
pythoninput_ids = tokenizer(text, return_tensors="pt").input_ids
with torch.no_grad():
    outputs = gpt2(input_ids)
    logits = outputs.logits  # shape: (batch_size, seq_len, vocab_size)

probs = F.softmax(logits[0, -1], dim=-1)   # softmax over the last position
topk = torch.topk(probs, 5)                # top 5 candidate next tokens


Copied from: Editing LLM-Project-1/README.md at main · VarunVoleti/LLM-Project-1 - <https://github.com/VarunVoleti/LLM-Project-1/edit/main/README.md>
```

### 2.4 Key takeaway

A language model isn't a black box — it's a composition of well-understood layers (attention + feed-forward) trained purely to predict the next token. At scale, this simple objective produces models with a surprisingly deep implicit understanding of language structure, meaning, and context.


## 3. Text Generation (Decoding)

Generation is fundamentally a loop:


Feed the current token sequence into the model.
Get a probability distribution over the next token.
A decoding strategy selects the next token from that distribution.
Append it and repeat.


Hugging Face's model.generate() handles this loop (plus stopping conditions, batching, etc.).

### 3.1 Greedy decoding

Always picks the single highest-probability token at each step.
```
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
output_tokens  # printing the output token IDs generated by the model
```

Downsides observed: repetition loops (e.g., "is is is…") and short-sighted token choices that can lead to incoherent longer-form text — the locally best token isn't always globally best.

### 3.2 Top-p (nucleus) sampling

Instead of always taking the top token, sample from the smallest set of tokens whose cumulative probability exceeds a threshold p (e.g., 0.9). This injects controlled randomness for more natural, less repetitive output while still staying coherent.
```
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
```

A reusable generate() helper was built to switch between "greedy" and "top_p" strategies and was run across multiple test prompts ("Once upon a time", "What is 2+2?", "Suggest a party theme.") to directly compare repetition (greedy) vs. variety (top-p).


## 4. From Simple GPT-2 to Modern LLMs

GPT-2 (2019, 124M params) does one thing: raw text completion. It has no notion of "answering a question" — given "What is 2+2?", it just continues the text as if it were part of a webpage.

Modern instruction-tuned models — here, Qwen3-0.6B (2025, ~600M params) — go through additional post-training stages that unlock:


Instruction following — interpreting input as a request, not just a text prefix to continue.
Multi-turn conversation — maintaining context across a dialogue.
Reasoning — some models "think" step-by-step using internal <think>...</think> reasoning tokens before producing a final answer.


### 4.1 Chat templates

Instruction-tuned models expect structured input, not raw text, e.g.:

<|user|>What is 2+2?<|assistant|>

Each model family defines its own format. tokenizer.apply_chat_template() handles this automatically:

pythonmessages = [{"role": "user", "content": prompt}]
chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

Without this step, even an instruction-tuned model just falls back to naive text completion.

### 4.2 GPT-2 vs. Qwen3-0.6B — side by side

Both models were loaded and given the same prompt ("What is 2+2?") using top-p sampling:

```

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

```
GPT-2 receives the raw prompt and blindly continues the text (no real "answer").
Qwen3-0.6B receives a properly chat-templated message and produces a direct, relevant response.


This side-by-side comparison makes the practical gap between "text completion" and "instruction-following chat model" concrete.


## 5. (Optional) Interactive LLM Playground

A small ipywidgets-based UI was built for hands-on experimentation directly in the notebook:
```
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

```

Prompt box (widgets.Textarea)
Model selector — GPT-2 vs. Qwen (widgets.Dropdown)
Decoding strategy selector — Greedy / Top-K / Top-P
Temperature slider (widgets.FloatSlider, range 0.1–2.0)
Generate button that triggers a generate_text() helper function, disables itself while running, and displays results as Markdown in an Output widget.


Key implementation detail: generate_text() automatically applies the chat template only for the instruction-tuned model (checked via a "qwen" in model_key.lower() condition), and strips the prompt tokens from the output before decoding so only the newly generated text is shown.


## 6. Inference Engines: Ollama, vLLM, SGLang

Up to this point, models were loaded directly in Python via transformers — great for learning, but not how production systems work. In practice, models run as servers: loaded once into memory/GPU, then exposed via an HTTP API that client applications call repeatedly.

EngineBest forOllamaEasy local use and experimentationvLLMHigh-throughput production servingSGLangFast serving + structured output

Most inference engines expose an OpenAI-compatible API, meaning the same openai Python client library can target any of these backends interchangeably — Ollama locally during development, then vLLM/SGLang in production. This pattern is explored further in a later project.


## Key Concepts Recap


Tokenization: text → integer IDs. Word / character / subword(BPE) tradeoffs; BPE (via GPT-2 tokenizer and tiktoken) is the modern standard.
Logits: raw, unnormalized next-token scores of shape (batch_size, seq_len, vocab_size); softmax turns them into a probability distribution.
Parameters: every learned weight/bias in the model; GPT-2 Small = 124M, illustrating how memory footprint scales with model size.
Decoding strategies: greedy (deterministic, repetitive) vs. top-p/nucleus sampling (diverse, coherent) — controls how the next token is chosen from the probability distribution.
Base vs. instruction-tuned models: GPT-2 only completes text; Qwen3-0.6B follows instructions and holds conversations thanks to additional post-training and chat templating.
Inference engines: Ollama / vLLM / SGLang serve models as APIs for production use, decoupling model loading from request handling.


### Tech Stack

**Python · PyTorch · Hugging Face transformers · tiktoken · ipywidgets · GPT-2 (124M) · Qwen3-0.6B**

## Learning Objectives Covered


 Understand tokenization and how raw text becomes discrete tokens
 Load and inspect a pretrained LLM (GPT-2) using Hugging Face
 Understand logits, probabilities, and next-token prediction
 Count and explore model parameters to understand model scale
 Explore decoding strategies: greedy decoding and top-p (nucleus) sampling
 See the leap from GPT-2 (text completion) to a modern instruction-following/reasoning model
 Get an overview of inference engines used to serve LLMs in production
