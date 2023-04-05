import nltk
import re

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

lem = WordNetLemmatizer()

def preProcessInputs(sen):
    var = re.sub('[^a-zA-Z]', ' ', sen) # substitute all characters other than a-z and A-Z with a space
    var = var.lower() # to lower case
    var = nltk.word_tokenize(var)
    var = [lem.lemmatize(word) for word in var if word not in set(stopwords.words('english'))] #lemmatize
    var = ' '.join(var)
    return var

def makeLabelsFile(outputs, fileName):
    labels = set(outputs)
    fwrite = open('nlu\\utils\\{}.txt'.format(fileName), 'w', encoding='utf-8')

    for label in labels:
        fwrite.write(label + '\n')
    fwrite.close()