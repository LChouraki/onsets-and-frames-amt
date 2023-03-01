
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pyaudio
import rtmidi
import numpy as np
import time
import argparse
import queue
from realtime_ar.mic_stream import MicrophoneStream
from threading import Thread
from autoregressive import *
from realtime_ar.transcribe import load_model, OnlineTranscriber

CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = pyaudio.PyAudio().get_default_input_device_info()['maxInputChannels']
RATE = SAMPLE_RATE



def get_buffer_and_transcribe(model, q):
    midiout = rtmidi.MidiOut()
    available_ports = midiout.get_ports()
    print(available_ports)
    if available_ports:
        midiout.open_port(0)
    else:
        midiout.open_virtual_port("My virtual output")

    start_flag = False
    pitches = []
    intervals = []
    curr_frame = 0
    transcriber = OnlineTranscriber(model)
    with MicrophoneStream(RATE, CHUNK, CHANNELS) as stream:
        audio_generator = stream.generator()
        on_pitch = []
        while True:
            data = stream._buff.get()
            decoded = np.frombuffer(data, dtype=np.int16) / 32768.0

            if np.max(abs(decoded)) > 0.1 and not start_flag:
                print("START")
                start_flag = True
            if start_flag:
                if CHANNELS > 1:
                    decoded = decoded.reshape(CHANNELS, -1)
                    decoded = np.mean(decoded, axis=0)
                frame_output = transcriber.inference(decoded)

                on_pitch += frame_output[1]
                for pitch in frame_output[1]:
                    note_on = [0x90, pitch + MIN_MIDI, 64]
                    pitches.append(pitch + MIN_MIDI)
                    intervals.append([curr_frame, curr_frame])
                    midiout.send_message(note_on)
                for pitch in frame_output[2]:
                    note_off = [0x90, pitch + MIN_MIDI, 0]
                    pitch_count = on_pitch.count(pitch)
                    [midiout.send_message(note_off) for i in range(pitch_count)]
                    pitch_idx = pitches[::-1].index(pitch + MIN_MIDI)
                    intervals[len(intervals) - 1 - pitch_idx][1] = curr_frame
                curr_frame += 0.032
                on_pitch = [x for x in on_pitch if x not in frame_output[2]]
                q.put(frame_output[0])
                if curr_frame > 10:
                    print("SAVING")
                    start_flag = False
                    save_midi('./test.mid', pitches, intervals, [100] * len(pitches))


def draw_plot(q):
    piano_roll = np.zeros((MAX_MIDI - MIN_MIDI + 1, 64 ))
    piano_roll[30, 0] = 1 # for test

    plt.ion()
    fig, ax = plt.subplots()

    plt.show(block=False)
    img = ax.imshow(piano_roll)
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
        if num_updated == 1:
            new_roll[:, :-1] = piano_roll[:,1:]
            new_roll[:, -1] = updated_frames[0]
        else:
            new_roll[:, :-num_updated] = piano_roll[:,num_updated:]
            # new_roll[:, -num_updated] = frame_output
            new_roll[:, -num_updated:] = np.asarray(updated_frames).T
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
    # print('model is running')
    draw_plot(q)
    # print("* done recording")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, default='../model-26000.pt')
    args = parser.parse_args()

    main(args.model_file)

