from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


digits = load_digits()
X = digits.data  
y = digits.target 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)


print("Accuracy:", clf.score(X_test, y_test))
