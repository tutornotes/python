#1. Tokenize sentence

import nltk
nltk.download('punkt_tab')  
a='I am going to kathmandu'
result=nltk.word_tokenize(a)
print(result)




#2. Tokenize Paragraph

import nltk
nltk.download('punkt_tab')
a="""Once, there was a hare who was best friends with a tortoise. The hare was very proud of how fast he could run, so one day, he challenged the tortoise to a race. The tortoise agreed, even though everyone thought he was way too slow to win. The race began, and the hare raced so fast that he was far ahead of the tortoise."""
result=nltk.word_tokenize(a)
print(result)




#3. Tokenize User Input


import nltk
nltk.download('punkt_tab')
a=input("Enter a sentence:- ")
result=nltk.word_tokenize(a)
print(result)



#4.Tokenize User Input using function

from nltk import word_tokenize
txt=input("Enter a sentence:- ")
def tokenize(str1):
  print(word_tokenize(str1))
tokenize(txt)




#2.	Find the word frequency



import nltk
nltk.download('punkt_tab')
t="""Once, there was a hare who was best friends with a tortoise. The hare was very proud of how fast he could run, so one day, he challenged the tortoise to a race. The tortoise agreed, even though everyone thought he was way too slow to win. The race began, and the hare raced so fast that he was far ahead of the tortoise."""
t1=nltk.word_tokenize(t)
print(t1)




count=[]
for i in t1:
  if i not in count:
    count.append(i)
for j in range(0,len(count)):
  print(count[j],t1.count(count[j]))


 
#3.	Demonstrate a bigram language model


from nltk import word_tokenize
import nltk
nltk.download('punkt_tab')   #for jupyter notebook use "nltk.download('punkt')"
sentence="She will be showing a demo of the company's new alarm system. a demo version of the software I saw a demo on how to use the computer program"
gram=2
token=word_tokenize(sentence)
bigram=[]
for i in range(len(token)-(gram-1)):
  temp=[token[j] for j in range(i,i+gram)]
  bigram.append(" ".join(temp))
print(bigram)


#4.	Demonstrate a trigram language model


import nltk
nltk.download('punkt_tab')
from nltk import ngrams
from nltk.tokenize import word_tokenize

sentence="She will be showing a demo of the company's new alarm system. a demo version of the software I saw a demo on how to use the computer program"
tokens=word_tokenize(sentence)
bigrams=list(ngrams(tokens,2))
trigrams=list(ngrams(tokens,3))
print("Bigrams: ",bigrams)
print("Trigrams: ",trigrams)


 
#5.	Generate regular expression for a given text


import re
text="Please contact support@gmail.com or sales+@amazon.in."
email_pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
regex=re.compile(email_pattern)
matches=regex.findall(text)
for match in matches:
  print(match)


#6.	Text Normalization


import re
import unicodedata
import nltk
def abc(text):
  normalized_text=text.lower()
  normalized_text=re.sub(r'[^\w\s]','',normalized_text)
  normalized_text=unicodedata.normalize('NFKD',normalized_text).encode('ASCII','ignore').decode('utf-8')
  normalized_text="".join(normalized_text.split())
  return normalized_text
input_text=input("Enter text to normalize:- ")
normalized_result=abc(input_text)
print("Normalize text: ",normalized_result)



















