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