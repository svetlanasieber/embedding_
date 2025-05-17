def predict_next(context_words):
    emb1 = embeddings[word2idx[context_words[0]]]
    emb2 = embeddings[word2idx[context_words[1]]]
    x = np.concatenate([emb1, emb2])
    logits = np.dot(x, W)
    probs = np.exp(logits - np.max(logits))
    probs = probs / np.sum(probs)
    idx = np.argmax(probs)
    return idx2word[idx], probs[idx]

print("I love ->", predict_next(["I", "love"]))
print("You love ->", predict_next(["You", "love"]))
