context = ["public", "static", "void"]
target = "main"

context_vec = embed(context)

predicted = model.predict(context_vec)

loss = loss_function(predicted, target)

model.update(loss)
