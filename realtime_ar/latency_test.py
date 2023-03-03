import pyaudio
import rtmidi
import numpy as np
import argparse
from realtime_ar.mic_stream import MicrophoneStream
from threading import Thread
from realtime_ar.transcribe import load_model, OnlineTranscriber
from autoregressive.midi import save_midi
from constants import *

CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = pyaudio.PyAudio().get_default_input_device_info()['maxInputChannels']
RATE = SAMPLE_RATE


def get_buffer_and_transcribe(model):
    midiout = rtmidi.MidiOut()
    available_ports = midiout.get_ports()
    print(available_ports)
    if available_ports:
        midiout.open_port(0)
    else:
        midiout.open_virtual_port("My virtual output")

    start_flag = False
    save = True
    pitches = []
    intervals = []
    curr_frame = 0
    transcriber = OnlineTranscriber(model)
    with MicrophoneStream(RATE, CHUNK, 6, CHANNELS) as stream:
        while True:
            data = stream._buff.get()
            decoded = np.frombuffer(data, dtype=np.int16) / 32768.0

            if np.max(abs(decoded)) > 1e-5 and not start_flag:
                print("START")
                start_flag = True
            if start_flag:
                if CHANNELS > 1:
                    decoded = decoded.reshape(CHANNELS, -1)
                    decoded = np.mean(decoded, axis=0)
                frame_output, onsets, offsets = transcriber.inference(decoded)

                for pitch in onsets:
                    note_on = [0x90, pitch + MIN_MIDI, 64]
                    midiout.send_message(note_on)
                    pitches.append(pitch + MIN_MIDI)
                    intervals.append([curr_frame, curr_frame])
                for pitch in offsets:
                    note_off = [0x90, pitch + MIN_MIDI, 0]
                    midiout.send_message(note_off)
                    if pitch + MIN_MIDI in pitches:
                        pitch_idx = pitches[::-1].index(pitch + MIN_MIDI)
                        intervals[len(pitches) - 1 - pitch_idx][1] = curr_frame
                curr_frame += 0.032

            if curr_frame > 15 and save:
                print("SAVING")
                save = False
                save_midi('./test.mid', pitches, intervals, [100] * len(pitches))


def main(model_file):
    model = load_model(model_file)

    print("* recording")
    t1 = Thread(target=get_buffer_and_transcribe, name=get_buffer_and_transcribe, args=(model,))
    t1.start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default='../model-3cnn.pt')
    args = parser.parse_args()

    main(args.model_file)

