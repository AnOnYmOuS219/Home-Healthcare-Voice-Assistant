import pyaudio
import json
import pyttsx3
import yaml

from nlu.classifier import classify
from vosk import Model, KaldiRecognizer

data = yaml.safe_load(open('nlu\\train.yml').read())

# Speech Recognition
model = Model("model")
rec = KaldiRecognizer(model, 16000)

# Speech Synthesis
engine = pyttsx3.init()

def evaluate(text):
    output = classify(text)
    entity = output['entity']
    conf = float(output['conf'])

    print('You said: {}  Confidence: {}'.format(text, conf))

    # if conf < 0.9:
    #     speak("Sorry, I didn't get you!")
    #     return
    
    idx = -1
    for i in range(len(intents)):
        if intents[i] == entity:
            idx = i
            break
    
    if idx == -1:
        speak("Sorry, I didn't get you!")
    else:
        speak(responses[idx])

def speak(text):
    engine.say(text)
    engine.runAndWait()

def initMicrophone():
    # Opens microphone for listening.
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
    stream.start_stream()
    return stream

intents, responses = [], []

for command in data['nlu']:
    intents.append(command['intent'])
    responses.append(command['response'])

stream = initMicrophone()

while True:
    data = stream.read(4000, exception_on_overflow = False)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        result = rec.Result()
        result = json.loads(result)
        evaluate(result['text'])