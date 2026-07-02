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