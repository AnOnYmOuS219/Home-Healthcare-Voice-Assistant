import pyttsx3 as tts
import sys
import yaml
import speech_recognition

from neuralintents import GenericAssistant
from nlu.classifier import classify

data = yaml.safe_load(open('nlu\\train.yml').read())

recognizer = speech_recognition.Recognizer()

engine = tts.init()
engine.setProperty('rate', 150)

def greet():
    speak("Hi, What can I do for you?")

def quit():
    speak("Bye")
    sys.exit(0)

def recognize_disease():
    global recognizer

    speak("Can you tell me any one symptom that you feel?")

    done = False

    while not done:
        try:
            with speech_recognition.Microphone() as mic:
                recognizer.adjust_for_ambient_noise(mic, duration=0.2)
                audio = recognizer.listen(mic)

                symptom = recognizer.recognize_google(audio)
                symptom = symptom.lower()
                idx = evaluate(symptom)

                if idx == -1:
                    speak("Sorry, I didn't get you!")
                else:
                    speak(responses[idx])
                    done = True
        except speech_recognition.UnknownValueError: 
            recognizer = speech_recognition.Recognizer()
            speak("Sorry, I didnt't get you! Please try again")

def evaluate(text):
    output = classify(text)
    entity = output['entity']
    conf = float(output['conf'])

    print(entity)
    print('You said: {}  Confidence: {}'.format(text, conf))
    
    idx = -1
    for i in range(len(intents)):
        if intents[i] == entity:
            idx = i
            break
    
    return idx

def speak(text):
    engine.say(text)
    engine.runAndWait()

mappings = {
    "greeting": greet,
    "quit": quit,
    "recognize_disease": recognize_disease
}

assistant = GenericAssistant('nlu/utils/intents.json', intent_methods=mappings)
# assistant.train_model()
# assistant.save_model()
assistant.load_model(model_name="assistant_model")

intents, responses = [], []

for command in data['nlu']:
    intents.append(command['intent'])
    responses.append(command['response'])

while True:
    try:
        with speech_recognition.Microphone() as mic:
            recognizer.adjust_for_ambient_noise(mic, duration=0.2)
            audio = recognizer.listen(mic)

            msg = recognizer.recognize_google(audio)
            msg = msg.lower()
            
            assistant.request(msg)

    except speech_recognition.UnknownValueError: 
        recognizer = speech_recognition.Recognizer()