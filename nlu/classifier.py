import numpy as np
import tensorflow as tf

from tensorflow import keras
from keras.models import load_model
from nlu.utils.multi_classification_utils import preProcessInputs

labels = open('nlu\\utils\\entities.txt', 'r', encoding='utf-8').read().split('\n')
model = load_model('nlu\\model\\model.h5')

label2idx = {}
idx2label = {}

for k, label in enumerate(labels):
    label2idx[label] = k
    idx2label[k] = label

def classify(text):
    x = np.zeros((1, 75, 256), dtype='float32')
    print("You said: ", text)
    
    text = preProcessInputs(text)
    print("Pre-processed text: ", text)

    if len(text) > 75:
        text = text[:75]

    for k, ch in enumerate(bytes(text.encode('utf-8'))):
        x[0, k, int(ch)] = 1.0

    out = model.predict(x)
    idx = out.argmax()

    return {"entity" : idx2label[idx], "conf" : max(out[0])}


labels_remedies = open('nlu\\utils\\entities_remedies.txt', 'r', encoding='utf-8').read().split('\n')
model_remedies = load_model('nlu\\model\\model_remedies.h5')

label2idx_r = {}
idx2label_r = {}

for k, label in enumerate(labels_remedies):
    label2idx_r[label] = k
    idx2label_r[k] = label

def classify_remedies(text):
    x = np.zeros((1, 75, 256), dtype='float32')
    print("You said: ", text)

    text = preProcessInputs(text)
    print("Pre-processed text: ", text)

    if len(text) > 75:
        text = text[:75]

    for k, ch in enumerate(bytes(text.encode('utf-8'))):
        x[0, k, int(ch)] = 1.0

    out = model_remedies.predict(x)
    idx = out.argmax()

    return {"entity" : idx2label_r[idx], "conf" : max(out[0])}