
#pip install -U pip setuptools wheel
#pip install spacy
#1.	Find the synonym of a word using WordNet


import nltk
nltk.download('wordnet',quiet=True)
from nltk.corpus import wordnet
word="Happy"
synonyms=[]
for syn in wordnet.synsets(word):
    for lemma in syn.lemmas():
        synonyms.append(lemma.name())
synonyms=list(set(synonyms))
print("Synonyms for",word+";")
print(synonyms)


#2.	Find the antonym of a word

import nltk
nltk.download('wordnet',quiet=True)
from nltk.corpus import wordnet
word="Good"
antonyms=[]
for syn in wordnet.synsets(word):
   for lemma in syn.lemmas():
       for antonym in lemma.antonyms():
           antonyms.append(antonym.name())
antonyms=list(set(antonyms))
print("Antonyms for",word+";")
print(antonyms) 


#3.	Implement semantic role labeling to identify named entities

import nltk
import spacy
nltk.download("averaged_perceptron_tagger")
nltk.download('words') # Corrected 'word' to 'words'
nlp=spacy.load("en_core_web_sm")
text="apple lnc. was founded by  steve jobs and steven worinak in capertion, california"
doc=nlp(text)
entities=[(ent.text,ent.label_) for ent in doc.ents]
for entity in entities:
    print(f"Entity:{entity[0]},Label:{entity[1]}")


#4.	Resolve the ambiguity

import nltk
import spacy
nlp=spacy.load("en_core_web_sm")
text="the chicken is ready too eat "
doc=nlp(text)
for token in doc:
    print(f"Token:{token.text},POS:{token.pos_},Sense:{token.lemma_}")

 
#5.	Translate the text using First-order logic

#!pip install pyDatalog


from pyDatalog import pyDatalog
pyDatalog.create_terms('X, human, mortal')
+human('John')
+human('Alice')
mortal(X) <= human(X)
print("Known mortals:")
print(mortal(X))
humans = ['John', 'Alice']
if all(mortal(x) == [(x,)] for x in humans):
    print("All humans are mortal")
else:
    print("Not all humans are mortal")



from pyDatalog import pyDatalog
pyDatalog.clear()
pyDatalog.create_terms('X, Y, teaches, students_of, younger_than')
+teaches('plato', 'aristotle')
+teaches('socrates', 'plato')
students_of(Y, X) <= teaches(X, Y)
younger_than(X, Y) <= students_of(Y, X)
print("Is Aristotle younger than Plato?")
print(younger_than('aristotle', 'plato'))
print("\nWho is a student of Socrates?")
print(students_of(X, 'socrates'))


