import torch
import librosa
from mel import melspectrogram
from decoding import extract_notes
from autoregressive.midi import save_midi
from constants import *

audio_file = "/Users/louischouraki/Documents/onsets-and-frames-pytorch/" \
        "data/GuitarSet/audio/audio_mono-pickup_mix/05_BN2-166-Ab_solo_mix.wav"

model = torch.load("./model-18000.pt", map_location=torch.device('cpu'))

audio, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
audio = torch.from_numpy(audio)

data = melspectrogram(audio.reshape(-1, audio.shape[-1])[:, :-1]).transpose(-1, -2)

result = model(data)
result = torch.softmax(result, dim=-1)
result = torch.argmax(result, dim=-1).squeeze()

test = result.numpy()
pitches, intervals = extract_notes(result >= 3, result > 1)

save_midi('./test.mid', pitches + MIN_MIDI, intervals * HOP_LENGTH / SAMPLE_RATE, [100] * len(pitches))
print()