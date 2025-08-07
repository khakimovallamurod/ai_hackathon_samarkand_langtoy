from pydub import AudioSegment
from vosk import Model, KaldiRecognizer
import wave
import json
import os
from google import genai
import config
import text_to_speach

def convert_ogg_to_wav(ogg_path, wav_path):
    audio = AudioSegment.from_file(ogg_path, format="ogg")
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(wav_path, format="wav", parameters=["-acodec", "pcm_s16le"])

def transcribe_vosk(wav_path):
    wf = wave.open(wav_path, "rb")
    model = Model("models/vosk-model-small-uz-0.22")
    rec = KaldiRecognizer(model, wf.getframerate())

    results = []

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            results.append(json.loads(rec.Result()))

    results.append(json.loads(rec.FinalResult()))
    result_texts = ""
    for res in results:
        if "text" in res:
            result_texts += res["text"] + " "
    return result_texts

def gemini_pompt(text):
    client = genai.Client(api_key=config.get_api_key())

    response = client.models.generate_content(
        model="gemini-2.0-flash", contents = f"""
        '{text}' \n\n Please answer this question briefly and in English.
        """
    )
    return response.text.replace('*', '')

if __name__ == '__main__':
    ogg_file = "tests/audio_2025-08-06_14-17-21.ogg"
    wav_file = "converted.wav"

    convert_ogg_to_wav(ogg_file, wav_file)
    text = transcribe_vosk(wav_file)
    print(text)
    gemini_answer = gemini_pompt(text)
    text_to_speach.text_to_speach_by_lang(text=gemini_answer, filename='test_voice01.mp3')
