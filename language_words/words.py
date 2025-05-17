import numpy as np

vocab_size = len(word2idx)
embed_size = 4 


embeddings = np.random.randn(vocab_size, embed_size)


W = np.random.randn(embed_size * 2, vocab_size)  

lr = 0.1 
