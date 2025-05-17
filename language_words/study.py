for epoch in range(300):  
    total_loss = 0
    for context, target in training_data:
      
        emb1 = embeddings[context[0]]
        emb2 = embeddings[context[1]]
        x = np.concatenate([emb1, emb2])  

        logits = np.dot(x, W) 

        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        loss = -np.log(probs[target])
        total_loss += loss

        grad_logits = probs
        grad_logits[target] -= 1 
        grad_W = np.outer(x, grad_logits)

        W -= lr * grad_W

        grad_x = np.dot(W, grad_logits)
        embeddings[context[0]] -= lr * grad_x[:embed_size]
        embeddings[context[1]] -= lr * grad_x[embed_size:]
        
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, loss {total_loss:.2f}")
