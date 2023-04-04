import yaml
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
from keras.layers import Dropout
from utils.multi_classification_utils import preProcessInputs, makeLabelsFile

def deepLearningModel(sent_max_len):
    model = Sequential()
    # model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dense(len(labels), activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    model.build((None, sent_max_len, 256))
    print(model.summary())
    return model

# load train data
data = yaml.safe_load(open('nlu\\train.yml').read())

# inputs is for training examples and outputs is for the problem tags
inputs, outputs = [], []

for command in data['nlu']:
    for example in command['examples']:
        inputs.append(preProcessInputs(example))

    for x in range(len(command['examples'])):
        outputs.append(command['intent'])

# calculate max length of an array (<=75)
max_sent = max([len(bytes(x.encode('utf-8'))) for x in inputs])

# matrix
input_data = np.zeros((len(inputs), max_sent, 256), dtype='float32')

for i, inp in enumerate(inputs):
    for k, ch in enumerate(bytes(inp.encode('utf-8'))):
        input_data[i, k, int(ch)] = 1.0

makeLabelsFile(outputs)

labels = open('nlu\entities.txt', 'r', encoding='utf-8').read().split('\n')

label2idx = {}
idx2label = {}

for k, label in enumerate(labels):
    label2idx[label] = k
    idx2label[k] = label

output_data = []

for output in outputs:
    output_data.append(label2idx[output])

# encoded output categorical data
output_data = to_categorical(output_data, len(labels))

model = deepLearningModel(75)
model.fit(input_data, output_data, epochs=256, batch_size=32)
model.save('nlu\model.h5')