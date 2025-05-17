data = [
    ([5, 3], 1),
    ([2, 4], 0),
    ([9, 8], 1),
    ([1, 6], 0),
]


w1, w2, b = 0.0, 0.0, 0.0
lr = 0.1 

for epoch in range(10):
    for (x, label) in data:
  
        y_pred = w1 * x[0] + w2 * x[1] + b
        y_pred = 1 if y_pred > 0 else 0  
        error = label - y_pred
       
        w1 += lr * error * x[0]
        w2 += lr * error * x[1]
        b  += lr * error * 1

print("w1, w2, b:", w1, w2, b)
