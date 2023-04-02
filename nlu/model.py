import yaml
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers import Bidirectional
import nltk
import re
nltk.download('all')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

data = yaml.safe_load(open('nlu\\train.yml').read())

inputs, outputs = [], []
lem = WordNetLemmatizer()

def preProcessInputs(sen):
    var = re.sub('[^a-zA-Z]', ' ', sen) # substitute all characters other than a-z and A-Z with a space
    var = var.lower() # to lower case
    var = nltk.word_tokenize(var)
    var = [lem.lemmatize(word) for word in var if word not in set(stopwords.words('english'))] #lemmatize
    var = ' '.join(var)
    return var


for command in data['nlu']:
    
    for example in command['examples']:
        inputs.append(preProcessInputs(example))

    for x in range(len(command['examples'])):
        outputs.append(command['intent'])

# print(inputs)
# print(outputs)

max_sent = max([len(bytes(x.encode('utf-8'))) for x in inputs])

input_data = np.zeros((len(inputs), max_sent, 256), dtype='float32')

for i, inp in enumerate(inputs):
    for k, ch in enumerate(bytes(inp.encode('utf-8'))):
        input_data[i, k, int(ch)] = 1.0

#output_data = to_categorical(output_data, len(output_data))

#print(input_data.shape)

# print(input_data[0].shape)

#print(len(chars))
#print('Max input seq:', max_sent)


labels = set(outputs)

fwrite = open('nlu\entities.txt', 'w', encoding='utf-8')
for label in labels:
    fwrite.write(label + '\n')
fwrite.close()

labels = open('nlu\entities.txt', 'r', encoding='utf-8').read().split('\n')

label2idx = {}
idx2label = {}

for k, label in enumerate(labels):
    label2idx[label] = k
    idx2label[k] = label

output_data = []

for output in outputs:
    output_data.append(label2idx[output])

output_data = to_categorical(output_data, len(labels))
# print(output_data)
embedding_vecor_length = 32
max_review_length = 75
top_words = 256

model = Sequential()
# model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
# model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(len(labels), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.build((None, 75, 256))
print(model.summary())
model.fit(input_data, output_data, epochs=512, batch_size=64)

model.save('nlu\model.h5')

# Classify any given text into a category of our NLU framework
def classify(text):
    # Create an input array
    x = np.zeros((1, max_sent, 256), dtype='float32')

    # Fill the x array with data from input text
    for k, ch in enumerate(bytes(text.encode('utf-8'))):
        x[0, k, int(ch)] = 1.0

    out = model.predict(x)

    #print('Text: "{}" is classified as "{}"'.format(text, idx2label[idx]))
    return idx2label