# Copyright 2022 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Experimental realtime_onsets transcription demo."""

import multiprocessing

import threading
import matplotlib.pyplot as plt
import time
import rtmidi
import torch

import audio_recorder
from transcribe import transcribe
from onsets_and_frames.constants import *
from onsets_and_frames.decoding import extract_notes
from absl import app
from absl import flags
import attr
from colorama import Fore
from colorama import Style
import numpy as np


flags.DEFINE_string('model_path', '../model-260000.pt',
                    'File path of TFlite model.')
flags.DEFINE_string('mic', '3', 'Optional: Input source microphone ID.')
flags.DEFINE_float('mic_amplify', 1.0, 'Multiply raw audio mic input')
flags.DEFINE_string(
    'wav_file', None,
    'If specified, will decode the first 10 seconds of this wav file.')
flags.DEFINE_integer(
    'sample_rate_hz', 16000,
    'Sample Rate. The model expects 16000. However, some microphones do not '
    'support sampling at this rate. In that case use --sample_rate_hz 48000 and'
    'the code will automatically downsample to 16000')
FLAGS = flags.FLAGS


class TfLiteWorker(multiprocessing.Process):
  """Process for executing TFLite inference."""

  def __init__(self, model_path, task_queue, result_queue):
    multiprocessing.Process.__init__(self)
    self._model_path = model_path
    self._task_queue = task_queue
    self._result_queue = result_queue
    self._model = None

  def setup(self):
    if self._model is not None:
      return

    self._model = torch.load(self._model_path, map_location=torch.device('cpu')).eval()

  def run(self):
    self.setup()
    while True:
      task = self._task_queue.get()
      if task is None:
        self._task_queue.task_done()
        return
      task(self._model)
      self._task_queue.task_done()
      self._result_queue.put(task)


@attr.s
class AudioChunk(object):
  serial = attr.ib()
  samples = attr.ib(repr=lambda w: '{} {}'.format(w.shape, w.dtype))


class AudioQueue(object):
  """Audio queue."""

  def __init__(self, callback, audio_device_index, sample_rate_hz,
               model_sample_rate, frame_length, overlap):
    # Initialize recorder.
    downsample_factor = sample_rate_hz / model_sample_rate
    self._recorder = audio_recorder.AudioRecorder(
        sample_rate_hz,
        downsample_factor=downsample_factor,
        device_index=audio_device_index)

    self._frame_length = frame_length
    self._overlap = overlap

    self._audio_buffer = np.zeros(2048, dtype=float)
    self._chunk_counter = 0
    self._callback = callback

  def start(self):
    """Start processing the queue."""
    with self._recorder:
      timed_out = False
      while not timed_out:
        assert self._recorder.is_active
        new_audio = self._recorder.get_audio(self._frame_length)
        self._audio_buffer[:-self._frame_length] = self._audio_buffer[self._frame_length:]
        self._audio_buffer[-self._frame_length:] = new_audio[0].squeeze() * 1

        self._callback(
            AudioChunk(self._chunk_counter,
                       self._audio_buffer))
        self._chunk_counter += 1
        '''audio_samples = np.concatenate(
            (self._audio_buffer, new_audio[0] * FLAGS.mic_amplify))

        # Extract overlapping
        first_unused_byte = 0
        for pos in range(0, audio_samples.shape[0] - self._frame_length,
                         self._frame_length - self._overlap):
          self._callback(
              AudioChunk(self._chunk_counter,
                         audio_samples[pos:pos + self._frame_length]))
          self._chunk_counter += 1
          first_unused_byte = pos + self._frame_length

        # Keep the remaining bytes for next time
        self._audio_buffer = audio_samples[first_unused_byte:]'''

# This actually executes in each worker thread!
class OnsetsTask(object):
  """Inference task."""

  def __init__(self, audio_chunk: AudioChunk):
    self.audio_chunk = audio_chunk
    self.result = None

  def __call__(self, model):

    samples = self.audio_chunk.samples
    print(samples)
    start = time.time()
    with torch.no_grad():
        results = transcribe(model, torch.from_numpy(samples).float())
    self.result = extract_notes(results['frame'], results['onset'], results['velocity'])[0]

    print(time.time() - start)


def draw_plot(q):
    piano_roll = np.zeros((MAX_MIDI - MIN_MIDI + 1, 64))
    piano_roll[30, 0] = 1 # for test

    plt.ion()
    fig, ax = plt.subplots()

    plt.show(block=False)
    img = ax.imshow(piano_roll)
    ax_background = fig.canvas.copy_from_bbox(ax.bbox)
    ax.invert_yaxis()
    fig.canvas.draw()

    midiout = rtmidi.MidiOut()
    available_ports = midiout.get_ports()
    if available_ports:
        midiout.open_port(0)
    else:
        midiout.open_virtual_port("My virtual output")

    while True:
        result = q.get()
        serial = result.audio_chunk.serial
        result_roll = result.result
        if serial > 0:
            result_roll = result_roll[3:]
        # num_updated = len(updated_frames)
        #if num_updated == 0:
        #    continue
        #last_frame = updated_frames[-1]
        frame_roll = np.zeros(MAX_MIDI - MIN_MIDI + 1)
        if len(result_roll) > 0:
            frame_roll[result_roll] = 1

        '''if num_updated == 0:
            continue
        if num_updated == 1:
            new_roll[:, :-1] = piano_roll[:,1:]
            new_roll[:, -1] = updated_frames[0]
        else:
            new_roll[:, :-num_updated] = piano_roll[:,num_updated:]
            # new_roll[:, -num_updated] = frame_output
            new_roll[:, -num_updated:] = np.asarray(updated_frames).T'''
        piano_roll[:, :-1] = piano_roll[:, 1:]
        piano_roll[:, -1] = frame_roll

        for pitch in result_roll:
            note_on = [0x90, pitch + 21, 64]
            midiout.send_message(note_on)
        fig.canvas.restore_region(ax_background)
        img.set_data(piano_roll)
        ax.draw_artist(img)
        fig.canvas.blit(ax.bbox)
        fig.canvas.flush_events()
        time.sleep(0.02)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    results = multiprocessing.Queue()

    #results_thread = threading.Thread(target=draw_plot, args=(results,))
    #results_thread.start()

    overlap_timesteps = 0
    overlap_wav = HOP_LENGTH * overlap_timesteps + 512

    tasks = multiprocessing.JoinableQueue()

    ## Make and start the workers
    num_workers = 4
    workers = [
        TfLiteWorker(FLAGS.model_path, tasks, results)
        for i in range(num_workers)
    ]
    for w in workers:
      w.start()

    audio_feeder = AudioQueue(
        callback=lambda audio_chunk: tasks.put(OnsetsTask(audio_chunk)),
        audio_device_index=FLAGS.mic if FLAGS.mic is None else int(FLAGS.mic),
        sample_rate_hz=int(FLAGS.sample_rate_hz),
        model_sample_rate=SAMPLE_RATE,
        frame_length=512,
        overlap=overlap_wav)

    audio_thread = threading.Thread(target=audio_feeder.start)
    audio_thread.start()
    draw_plot(results)


def console_entry_point():
  app.run(main)


if __name__ == '__main__':
  console_entry_point()
