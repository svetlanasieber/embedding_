import numpy as np
import spacy


nlp = spacy.load("en_core_web_md")

word1 = nlp("cat")
word2 = nlp("dog")
word3 = nlp("car")

print("cat/dog similarity:", word1.similarity(word2))
print("cat/car similarity:", word1.similarity(word3))
