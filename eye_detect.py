import pyttsx3
# import urllib.request
import requests

engine = pyttsx3.init()


class Myclass:
    def _int_(self):
        pass

    def Blink(self):
        if self == 1:
            engine.say("AC on")
            engine.runAndWait()
            print("AC on")


        if self == 2:
            engine.say("AC off")
            engine.runAndWait()
            print("AC off")


        if self == 3:
            engine.say("Fan on")
            engine.runAndWait()
            print("Fan on")


        if self == 4:
            engine.say("Fan off")
            engine.runAndWait()
            print("Fan off")


        if self == 5:
            engine.say("TV on")
            engine.runAndWait()
            print("TV on")


        if self == 6:
            engine.say("TV off")
            engine.runAndWait()
            print("TV off")