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


