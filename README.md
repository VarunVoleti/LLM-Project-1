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

bashcurl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt

Register as a Jupyter kernel if needed:

bashpython -m ipykernel install --user --name=llm_playground --display-name "llm_playground"

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





### 1.2 Character-level tokenization


Vocabulary = every individual character (letters + punctuation), each with its own ID.
Same encode/decode pattern, but operating character-by-character instead of word-by-word.
Trade-offs:

Solves the OOV problem (any text can be represented).
Produces much longer sequences (more tokens per input → higher compute cost).
Individual characters carry very little semantic meaning on their own, so the model has to work harder to learn relationships across many more steps.





### 1.3 Subword-level tokenization (BPE)


Byte-Pair Encoding (BPE), along with WordPiece and SentencePiece, is the standard approach in modern LLMs. It learns a vocabulary from data rather than using fixed rules:

Start with individual characters/bytes as tokens.
Count all adjacent token pairs across a large corpus.
Merge the most frequent pair into a new token.
Repeat until a target vocabulary size is reached (e.g., 50,000).



Example: "unbelievable" → ["un", "believ", "able"].
Used the pretrained GPT-2 tokenizer (AutoTokenizer.from_pretrained("gpt2")) via Hugging Face to see BPE applied in practice, using .encode(), .decode(), and .convert_ids_to_tokens().


### 1.4 tiktoken


tiktoken is OpenAI's fast, production-grade tokenizer, used to keep token counting consistent with how OpenAI models actually process text.
Compared GPT-2's tokenizer (tiktoken.get_encoding("gpt2")) against GPT-4's (cl100k_base) on the same sentence (including an emoji and a hyphenated compound word) to see how vocabulary design has evolved — GPT-4's tokenizer handles emojis/special characters more efficiently and produces different segmentations for the same input.


### 1.5 Key takeaways

MethodProsConsWord-levelSimple, intuitiveHuge vocab, OOV problemCharacter-levelHandles any textLong sequences, weak semantics per tokenSubword / BPEBalances vocab size & sequence lengthSlightly more complex to implementtiktokenProduction-grade, OpenAI-consistentN/A — used for inspection/comparison

BPE (or similar subword methods) is the default choice for essentially all modern LLMs.


## 2. What Is a Language Model?

At its core: a function that predicts the next token given a sequence of previous tokens. GPT-2 stacks multiple Transformer blocks, each combining:


Attention — mixes information between tokens in the sequence.
Feed-forward layers — transforms that mixed representation.


The final output is a set of logits — one raw score per vocabulary token — indicating how likely each token is to come next.

### 2.1 Loading GPT-2

pythonfrom transformers import GPT2LMHeadModel
gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")  # 124M parameter model
block = gpt2.transformer.h[0]  # first Transformer block

Inspecting gpt2.transformer.h[0] shows the internal structure of a single Transformer block (attention + MLP layers).

### 2.2 Counting parameters

pythontotal = sum(p.numel() for p in gpt2.parameters())
print(f"Total number of parameters in GPT-2: {total:,}")

GPT-2 Small has 124M parameters; GPT-4 is estimated at over 1 trillion. A useful mental exercise: at 16-bit precision (2 bytes/parameter), 124M params ≈ 248 MB just to store weights — scaling this to a 70B-parameter model implies ~140 GB, illustrating why large models require serious infrastructure (multi-GPU setups, quantization, etc.) just to load into memory.

### 2.3 From text to predictions (logits → probabilities)

pythoninput_ids = tokenizer(text, return_tensors="pt").input_ids
with torch.no_grad():
    outputs = gpt2(input_ids)
    logits = outputs.logits  # shape: (batch_size, seq_len, vocab_size)

probs = F.softmax(logits[0, -1], dim=-1)   # softmax over the last position
topk = torch.topk(probs, 5)                # top 5 candidate next tokens


Logits have shape (batch_size, seq_len, vocab_size) — one score vector per position, per possible vocabulary token.
Softmax converts the final position's logits into a probability distribution over the whole vocabulary.
torch.topk extracts the 5 most probable next tokens for inspection.


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

pythonoutput_tokens = model.generate(
    **input_tokens,
    max_new_tokens=10,
    do_sample=False,          # greedy
)

Downsides observed: repetition loops (e.g., "is is is…") and short-sighted token choices that can lead to incoherent longer-form text — the locally best token isn't always globally best.

### 3.2 Top-p (nucleus) sampling

Instead of always taking the top token, sample from the smallest set of tokens whose cumulative probability exceeds a threshold p (e.g., 0.9). This injects controlled randomness for more natural, less repetitive output while still staying coherent.

pythonoutput_tokens = model.generate(
    **input_tokens,
    max_new_tokens=10,
    do_sample=True,
    top_p=0.9,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id,
)

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


GPT-2 receives the raw prompt and blindly continues the text (no real "answer").
Qwen3-0.6B receives a properly chat-templated message and produces a direct, relevant response.


This side-by-side comparison makes the practical gap between "text completion" and "instruction-following chat model" concrete.


## 5. (Optional) Interactive LLM Playground

A small ipywidgets-based UI was built for hands-on experimentation directly in the notebook:


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
