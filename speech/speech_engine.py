# speech/speech_engine.py
import queue
import threading

import pyttsx3


class SpeechEngine:
    def __init__(self, rate=170, volume=1.0, voice_index=0):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)

        voices = self.engine.getProperty("voices")
        if voices and voice_index < len(voices):
            self.engine.setProperty("voice", voices[voice_index].id)

        self.queue = queue.Queue()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def speak(self, text):
        if not text:
            return
        self.queue.put(text)

    def _run(self):
        while True:
            text = self.queue.get()
            if text is None:
                break
            self.engine.say(text)
            self.engine.runAndWait()

    def shutdown(self):
        self.queue.put(None)
