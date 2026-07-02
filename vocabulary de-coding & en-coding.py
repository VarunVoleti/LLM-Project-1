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