import pyttsx3
import queue
import subprocess
import sys
import threading


class SpeechOutput:
    def __init__(self):
        self.speech_queue = queue.Queue()
        self._worker_error = None
        self._ready = threading.Event()

        # Run speech separately so webcam processing stays responsive
        self.worker = threading.Thread(
            target=self._speech_worker,
            daemon=True
        )
        self.worker.start()

        if not self._ready.wait(timeout=10):
            raise TimeoutError(
                "Text-to-speech engine did not initialize within 10 seconds."
            )

        self.raise_if_failed()

    def _speech_worker(self):
        engine = None
        comtypes = None
        com_initialized = False

        try:
            if sys.platform != "darwin":
                if sys.platform == "win32":
                    import comtypes
                    from comtypes.client import CreateObject

                    comtypes.CoInitialize()
                    com_initialized = True
                    engine = CreateObject("SAPI.SpVoice")
                    engine.Rate = 1
                    engine.Volume = 100
                else:
                    engine = pyttsx3.init()
                    engine.setProperty("rate", 160)

            self._ready.set()

            while True:
                text = self.speech_queue.get()

                try:
                    if text is None:
                        return

                    if sys.platform == "darwin":
                        subprocess.run(
                            ["say", "-r", "180", text],
                            check=True,
                        )
                    elif sys.platform == "win32":
                        engine.Speak(text)
                    else:
                        engine.say(text)
                        engine.runAndWait()
                except Exception as exc:
                    self._worker_error = exc
                    return
                finally:
                    self.speech_queue.task_done()
        except Exception as exc:
            self._worker_error = exc
            self._ready.set()
        finally:
            if com_initialized:
                comtypes.CoUninitialize()

    def raise_if_failed(self):
        if self._worker_error is not None:
            raise RuntimeError(
                "Text-to-speech failed. Check the audio output and "
                "pyttsx3 installation."
            ) from self._worker_error

        if not self.worker.is_alive():
            raise RuntimeError("Text-to-speech worker has stopped.")

    def speak(self, text):
        self.raise_if_failed()

        text = str(text).strip()

        if not text or text.upper() == "UNKNOWN":
            return

        self.speech_queue.put(text)

    def close(self):
        if self.worker.is_alive():
            self.speech_queue.put(None)
            self.worker.join()


if __name__ == "__main__":
    speaker = SpeechOutput()

    while True:
        message = input("Enter recognized text (or q to quit): ")

        if message.lower() == "q":
            break

        speaker.speak(message)
