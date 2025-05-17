from transformers import pipeline
from pydub import AudioSegment


def convert_mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")

def transcribe_audio(wav_path):
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")
    result = asr(wav_path)
    return result['text']

if __name__ == "__main__":
    mp3_file = "sample.mp3"
    wav_file = "sample.wav"

    print("🔄 Converting MP3 to WAV...")
    convert_mp3_to_wav(mp3_file, wav_file)

    print("🧠 Transcribing with Whisper model...")
    transcription = transcribe_audio(wav_file)

    print("\n📝 Transcription Result:")
    print(transcription)
