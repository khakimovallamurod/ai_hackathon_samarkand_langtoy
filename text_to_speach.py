from gtts import gTTS
import os
from gtts.lang import tts_langs

def get_supported_languages():
    languages = tts_langs()
    return languages

def text_to_speach_by_lang(text: str, lang: str = 'en', filename: str = 'output.mp3'):
    folder = 'results'
    if not os.path.exists(folder):
        os.makedirs(folder)
    tts = gTTS(text=text, lang=lang)
    tts.save(f'{folder}/{filename}')
    return tts

if __name__ == "__main__":
    my_text = "Welcome to Samarkand AI Hackathon!"
    text_to_speach_by_lang(my_text)    

