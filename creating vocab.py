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