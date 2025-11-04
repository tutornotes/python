

#1.	Implement RNN for sequence labeling

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

vocab = {'I': 0, 'love': 1, 'natural': 2, 'language': 3, 'processing': 4, 'like': 5, 'deep': 6, 'learning': 7}

sequences = [['I', 'love', 'natural', 'language', 'processing']]
labels = [['PRON', 'VERB', 'ADJ', 'NOUN', 'NOUN']]

sequence_indices = [[vocab[word] for word in sequence] for sequence in sequences]

label_vocab = {'PRON': 0, 'VERB': 1, 'ADJ': 2, 'NOUN': 3}
label_indices = [[label_vocab[label] for label in label_sequence] for label_sequence in labels]

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        output = self.fc(output)
        return output

input_size = len(vocab)
hidden_size = 64
output_size = len(label_vocab)
model = RNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    inputs = torch.tensor(sequence_indices).long()
    labels = torch.tensor(label_indices).view(-1).long()

    outputs = model(inputs)
    outputs = outputs.view(-1, output_size)

    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    print(f'Epoch[{epoch+1}/{num_epochs}], loss:{loss.item():.4f}')

with torch.no_grad():
    test_sequence = [['I', 'like', 'deep', 'learning']]
    test_sequence_indices = [[vocab[word] for word in sequence] for sequence in test_sequence]
    inputs = torch.tensor(test_sequence_indices).long()
    outputs = model(inputs)
    predicted_labels = torch.argmax(outputs, dim=2)
    
    index_to_label = {v: k for k, v in label_vocab.items()}
    
    predicted_labels = [[index_to_label[label.item()] for label in sequence] for sequence in predicted_labels]
    print(f'Predicted Labels:{predicted_labels}')

 
 
#2.	Implement POS tagging using LSTM


import spacy
import tensorflow as tf
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
nlp = spacy.load("en_core_web_sm")

text = "The quick brown fox jumps over the lazy dog"

doc = nlp(text)

tokens = [token.text for token in doc]
pos_tags = [token.pos_ for token in doc]

label_encoder = LabelEncoder()
pos_labels = label_encoder.fit_transform(pos_tags)

X_train, X_test, Y_train, Y_test = train_test_split(tokens, pos_labels, test_size=0.2, random_state=42)

tokenizer = keras.layers.TextVectorization()
tokenizer.adapt(X_train)

X_train = tokenizer(X_train)
X_test = tokenizer(X_test)
Y_train = Y_train.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)

model = keras.Sequential([
    keras.layers.Embedding(input_dim=len(tokenizer.get_vocabulary()), output_dim=128, mask_zero=True),
    keras.layers.LSTM(128, return_sequences=False),
    keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=5, validation_split=0.2)
loss, accuracy = model.evaluate(X_test, Y_test)
print(f'loss:{loss}, accuracy:{accuracy}')



 
#3.	Implement Named Entity Recognizer

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('maxent_ne_chunker_tab')
text = "Apple Inc. is an American multinational technology company headquartered in Cupertino,California."

tokens = nltk.word_tokenize(text)

pos_tags = nltk.pos_tag(tokens)

named_entities = nltk.chunk.ne_chunk(pos_tags)

entities = []
for subtree in named_entities:
    if isinstance(subtree, nltk.Tree):
        entity = " ".join([word for word, tag in subtree.leaves()])
        label = subtree.label()
        entities.append((entity, label))
for entity, label in entities:
    print(f'Entity:{entity},Label:{label}')



#4.	Word sense disambiguation by LSTM/GRU

import nltk
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize
nltk.download('all')
def get_semantic(seq, key_word):
    temp = word_tokenize(seq)
    temp = lesk(temp, key_word)
    return temp.definition()
keyword = 'book'
seq1 = 'I love reading books on coding.'
seq2 = 'The table was already booked by someone else.'

keyword1 = 'jam'
seq3 = 'My mother prepares very yummy jam.'
seq4 = 'signal jammers are the reason for no signal.'

print(get_semantic(seq1, keyword))
print(get_semantic(seq2, keyword))

print(get_semantic(seq3, keyword1))
print(get_semantic(seq4, keyword1))