
#1.	Perform Lemmatization


import nltk
nltk.download('punkt_tab')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')



lemmatizer = WordNetLemmatizer()
sen="The boys and girls were presented in classes."
words=nltk.word_tokenize(sen)
lemmatized_word=[lemmatizer.lemmatize(word)for word in words]
lemmatized_sen=' '.join(lemmatized_word)
print(lemmatized_sen)


 

#2.	Perform Stemming


import nltk
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
words=["running","files","jumping","quickly"]
stemmed_words=[stemmer.stem(word)for word in words]
for original,stemmed in zip(words,stemmed_words):
  print(f"{original}->{stemmed}")



#3.	Identify parts-of Speech using Penn Treebank tag set.

import nltk
nltk.download('averaged_perceptron_tagger_eng')
sentence="the cats are chasing mice"
words=nltk.word_tokenize(sentence)
pos_tags=nltk.pos_tag(words)
print(pos_tags)


 
#4.	Implement HMM for POS tagging

import nltk
import random
from nltk.tag import hmm
nltk.download('punkt')
nltk.download('treebank')
nltk.download('punkt_tab')

corpus = nltk.corpus.treebank

sentences = corpus.sents()
tagged_sentences = corpus.tagged_sents()

random.seed(123)  
split_ratio = 0.8
split_index = int(len(tagged_sentences) * split_ratio)

training_sentences = tagged_sentences[:split_index]
testing_sentences = tagged_sentences[split_index:]

trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(training_sentences)

accuracy = hmm_tagger.evaluate(testing_sentences)
print("HMM POS Tagger Accuracy:", accuracy)

new_sentence = "This is a test sentence"
new_words = nltk.word_tokenize(new_sentence)
predicted_tags = hmm_tagger.tag(new_words)

print("Predicted POS tags for the new sentence:")
print(predicted_tags)


 
#5.	Build a Chunker

import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng') 
L="The quick brown fox jumps over the lazy dog"
words=nltk.word_tokenize(L)
pos_tags=nltk.pos_tag(words)
grammer=r"""NP:{<DT|JJ|NN.*>+}"""
chunk_parcer=nltk.RegexpParser(grammer)
chunks_sentence=chunk_parcer.parse(pos_tags)
for subtree in chunks_sentence.subtrees():
  if subtree.label()=='NP':
    print(' '.join(word for word,tag in subtree.leaves()))


 

#6.	Summerization


import sumy
from sumy.parsers.plaintext import PlaintextParser 
from sumy.nlp.tokenizers import Tokenizer 
from sumy.summarizers.lsa import LsaSummarizer 
text=""" Text summarizer is the process of generating short,fluent and most importantly accurate summary  of respectevly longer text document""" # Corrected typo

parcer=PlaintextParser.from_string(text,Tokenizer("english"))
summarizer=LsaSummarizer()
sentences_count=int(input("enter the value"))
summary=summarizer(parcer.document,sentences_count)
for sentence in summary:
  print(sentence)
