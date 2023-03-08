
import matplotlib
import matplotlib.pyplot as plt
import pyaudio
import rtmidi
import numpy as np
import time
import argparse
import queue
from realtime_ar.mic_stream import MicrophoneStream
from threading import Thread
from realtime_ar.transcribe import load_model, OnlineTranscriber
from autoregressive.midi import save_midi
from constants import *

matplotlib.use('Qt5Agg')
CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = pyaudio.PyAudio().get_default_input_device_info()['maxInputChannels']
RATE = SAMPLE_RATE


def get_buffer_and_transcribe(model, q):

    transcriber = OnlineTranscriber(model)
    with MicrophoneStream(RATE, CHUNK, 1, CHANNELS) as stream:
        on_pitch = []
        while True:
            data = stream._buff.get()
            decoded = np.frombuffer(data, dtype=np.int16) / 32768.0

            if CHANNELS > 1:
                decoded = decoded.reshape(CHANNELS, -1)
                decoded = np.mean(decoded, axis=0)
            frame_output, onsets, offsets = transcriber.inference(decoded)

            '''if len(onsets) > 0:
                print("ONSET", onsets)
            if len(offsets) > 0:
                print("OFFSET", offsets)'''
            for pitch in onsets:
                note_on = [0x90, pitch + MIN_MIDI, 64]
                midiout.send_message(note_on)

            for pitch in offsets:
                note_off = [0x90, pitch + MIN_MIDI, 0]
                midiout.send_message(note_off)

            q.put(frame_output)


def draw_plot(q):
    piano_roll = np.zeros((MAX_MIDI - MIN_MIDI + 1, 64))
    piano_roll[30, 0] = 1  # for test

    plt.ion()
    fig, ax = plt.subplots()

    plt.show(block=False)
    img = ax.imshow(piano_roll, cmap=plt.colormaps['gnuplot2'])
    #ax.set_facecolor((1, 1, 1))
    ax_background = fig.canvas.copy_from_bbox(ax.bbox)
    ax.invert_yaxis()
    fig.canvas.draw()

    while True:
        updated_frames = []
        while q.qsize():
            updated_frames.append(q.get())
        num_updated = len(updated_frames)
        if num_updated == 0:
            continue
        new_roll = np.zeros_like(piano_roll)

        updated_frames = np.array(updated_frames)
        new_roll[:, :-1] = piano_roll[:, 1:]

        new_roll[:, -1:] = np.expand_dims(np.sum(updated_frames, axis=0) + 0.9 * new_roll[:, -2], axis=-1)

        piano_roll = new_roll
        fig.canvas.restore_region(ax_background)
        img.set_data(piano_roll)
        ax.draw_artist(img)
        fig.canvas.blit(ax.bbox)
        fig.canvas.flush_events()
        time.sleep(0.02)


def main(model_file):
    model = load_model(model_file)
    
    q = queue.Queue()
    print("* recording")
    t1 = Thread(target=get_buffer_and_transcribe, name=get_buffer_and_transcribe, args=(model, q))
    t1.start()

    draw_plot(q)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default='../model-24000.pt')
    args = parser.parse_args()

    midiout = rtmidi.MidiOut()
    available_ports = midiout.get_ports()
    print(available_ports)
    if available_ports:
        midiout.open_port(0)
    else:
        midiout.open_virtual_port("My virtual output")

    try:
        main(args.model_file)

    except KeyboardInterrupt:
        for pitch in range(MIN_MIDI, MAX_MIDI):
            note_off = [0x90, pitch + MIN_MIDI, 0]
            midiout.send_message(note_off)

