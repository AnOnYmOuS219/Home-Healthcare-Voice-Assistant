from vosk import Model, KaldiRecognizer
import pyaudio
import json
import pyttsx3
import yaml

# Import NLU classifier
from nlu.classifier import classify

data = yaml.safe_load(open('nlu\\train.yml').read())

intents, responses = [], []

for command in data['nlu']:
    
    # for example in command['examples']:
    intents.append(command['intent'])

    # for x in range(len(command['examples'])):
    responses.append(command['response'])

# print(intents)
# print(responses)

def evaluate(text):
    output = classify(text)

    entity = output['entity']
    conf = float(output['conf'])

    print('You said: {}  Confidence: {}'.format(text, conf))

    if conf < 0.9:
        return
    
    idx = -1
    print(len(intents))
    for i in range(len(intents)):
        # print(intents[i])
        if intents[i] == entity:
            idx = i
            break
    
    if idx == -1:
        speak("Sorry, I didn't get you!")
    else:
        speak(responses[idx])

# Speech Recognition
model = Model("model")
rec = KaldiRecognizer(model, 16000)

# Speech Synthesis
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

# speak("Howdy partners? Wassup? I am fine, yourself? Hey, I am here! Just tell me yours ....")

# Opens microphone for listening.
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
stream.start_stream()

while True:
    data = stream.read(4000, exception_on_overflow = False)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        # result is a string
        result = rec.Result()
        # convert it to a json/dictionary
        result = json.loads(result)

        print(result['text'])
        evaluate(result['text'])

