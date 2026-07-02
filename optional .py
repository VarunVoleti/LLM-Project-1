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
