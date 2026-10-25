"""Gnani STT/TTS layer (Gnani API only). Follows the capstone brief's SDK usage."""
import os, re, threading, time

_LOCK = threading.Lock()  # Gnani allows only 1 active request at a time


def _retry(fn, tries=4):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i < tries - 1 and ("429" in str(e) or "RATE_LIMITED" in str(e)):
                time.sleep(2 * (i + 1)); continue
            raise

_HINDI_WORDS = re.compile(r"\b(aap|aapka|aapki|hai|hain|nahi|kya|kitne|mera|mujhe|kar|karo|karna|paise|refund ho|ho jayega|kijiye|dhanyavaad|ji)\b", re.I)


def reply_language(text):
    """TTS language code from the reply text. Devanagari -> hi-IN; romanised Hinglish -> HINGLISH_TTS_LANG (default hi-IN); else en-IN."""
    if re.search(r"[\u0900-\u097F]", text):
        return "hi-IN"
    if len(_HINDI_WORDS.findall(text)) >= 2:
        return os.getenv("HINGLISH_TTS_LANG") or "hi-IN"
    return "en-IN"


def speakable(text):
    """Make text TTS-friendly: strip markdown/symbols, group 8-digit IDs, spell currency symbol."""
    t = text.replace("\u202f", " ").replace("\u00a0", " ").replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", ", ")
    t = re.sub(r"[*_`#>]+", "", t)
    t = re.sub(r"\b(\d{4})(\d{4})\b", r"\1 \2", t)
    t = re.sub(r"(?:₹|Rs\.?\s?)\s?([\d,]+)", lambda m: m.group(1).replace(",", "") + " rupees", t)
    return re.sub(r"\s+", " ", t).strip()


class Voice:
    RECOVERY = "Sorry, I didn't catch that. Could you say it again?"

    def __init__(self):
        from gnani.stt import GnaniSTTClient
        from gnani.tts import GnaniTTSClient
        self.stt, self.tts = GnaniSTTClient(), GnaniTTSClient()  # read GNANI_API_KEY from env
        self.model = os.getenv("GNANI_TTS_MODEL") or "timbre-v2.5"
        self.voice = os.getenv("GNANI_TTS_VOICE") or "Kaveri"

    def listen(self, audio_path, language_code="en-IN"):
        try:
            with _LOCK:
                r = _retry(lambda: self.stt.transcribe(str(audio_path), language_code=language_code))
            return (r.get("transcript") or "").strip()
        except Exception as e:
            print(f"[STT error] {audio_path}: {e}")
            return ""

    def speak(self, text, out_path, language=None):
        first = language or reply_language(text)
        chain = [first] + [l for l in ("auto", "en-IN") if l != first]
        spoken = speakable(text)
        for lang in chain:
            for attempt in range(2):
                try:
                    with _LOCK:
                        _retry(lambda: self.tts.synthesize(spoken, voice=self.voice, model=self.model, language=lang, output_file=str(out_path)))
                    return True
                except Exception as e:
                    print(f"[TTS error] lang={lang} attempt {attempt + 1}: {e}")
                    if "unsupported language" in str(e).lower():
                        break  # try the next language in the chain
        return False
