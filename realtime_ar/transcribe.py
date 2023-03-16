import torch
import torch as th
import numpy as np
from constants import *
from mel import melspectrogram
from utils import summary
import time

class OnlineTranscriber:
    def __init__(self, model, return_roll=True):
        self.model = model
        self.model.eval()
        for i in (0, 4, 10):
            self.model.acoustic_model.cnn[i].padding = (1, 1, 0, 0)
        '''for i in (0, 5): # change for 2cnn
            self.model.acoustic_model.cnn[i].padding = (0, 1)'''

        melspectrogram.stft.padding = False
        self.audio_buffer = th.zeros((1, FILTER_LENGTH + 6 * HOP_LENGTH)).to(th.float) # change for 2cnn
        self.mel_buffer = melspectrogram(self.audio_buffer)

        self.acoustic_layer_outputs = self.init_acoustic_layer(self.mel_buffer)

        self.hidden = model.init_lstm_hidden(1, 'cpu')
        self.prev_output = th.zeros((1, 1, MAX_MIDI - MIN_MIDI + 1 + 4)).to(th.long)
        self.buffer_length = 0
        self.sr = SAMPLE_RATE
        self.return_roll = return_roll
        self.cnt = -1
        self.inten_threshold = 0.05
        self.patience = 100
        self.num_under_thr = 0

    def update_buffer(self, audio):
        t_audio = th.tensor(audio).to(th.float)
        new_buffer = th.zeros_like(self.audio_buffer)
        new_buffer[0, :-len(t_audio)] = self.audio_buffer[0, len(t_audio):]
        new_buffer[0, -len(t_audio):] = t_audio
        self.audio_buffer = new_buffer

    def update_mel_buffer(self):
        self.mel_buffer[:, :, :-1] = self.mel_buffer[:, :, 1:]
        self.mel_buffer[:, :, -1:] = melspectrogram(self.audio_buffer[:, -FILTER_LENGTH:])

    def init_acoustic_layer(self, input_mel):
        x = input_mel.transpose(-1, -2).unsqueeze(1)
        acoustic_layer_outputs = []
        for i, layer in enumerate(self.model.acoustic_model.cnn):
            x = layer(x)
            if i in [3, 9]:
                acoustic_layer_outputs.append(x)
        return acoustic_layer_outputs

    def update_acoustic_out(self, mel):
        x = mel[:, -3:, :].unsqueeze(1)
        layers = self.model.acoustic_model.cnn
        for i in range(4):
            x = layers[i](x)
        self.acoustic_layer_outputs[0][:, :, :-1, :] = self.acoustic_layer_outputs[0][:, :, 1:, :]
        self.acoustic_layer_outputs[0][:, :, -1:, :] = x

        x = self.acoustic_layer_outputs[0][:, :, -3:, :]

        for i in range(4, 10):       # change for 2cnn
            x = layers[i](x)
        self.acoustic_layer_outputs[1][:, :, :-1, :] = self.acoustic_layer_outputs[1][:, :, 1:, :]
        self.acoustic_layer_outputs[1][:, :, -1:, :] = x

        x = self.acoustic_layer_outputs[1] # change for 2cnn
        for i in range(10, 16): # change for 2cnn
            x = layers[i](x)
        x = x.transpose(1, 2).flatten(-2)
        return self.model.acoustic_model.fc(x)
    
    def switch_on_or_off(self):
        pseudo_intensity = torch.max(self.audio_buffer) - torch.min(self.audio_buffer)
        if pseudo_intensity < self.inten_threshold:
            self.num_under_thr += 1
        else:
            self.num_under_thr = 0

    def inference(self, audio):
        # time_list = []

        with th.no_grad():
            self.update_buffer(audio)
            self.switch_on_or_off()
            if self.num_under_thr > self.patience:
                if self.return_roll:
                    return [0] * (MAX_MIDI - MIN_MIDI + 1 + 4), [], []
                else:
                    return [], []
            self.update_mel_buffer()

            acoustic_out = self.update_acoustic_out(self.mel_buffer.transpose(-1, -2))
            language_out, self.hidden = self.model.lm_model_step(acoustic_out, self.hidden, self.prev_output)
            out = language_out.argmax(dim=3)

        frame_out = out[0, 0].numpy().astype('float')
        onset_pitches = np.argwhere((frame_out >= 3))

        off_pitches = np.argwhere((frame_out <= 1) * (self.prev_output.numpy().squeeze() > 1))
        self.prev_output = out

        frame_out[frame_out < 3] = 0
        frame_out[frame_out >= 3] = 0.4

        return frame_out, onset_pitches, off_pitches


def load_model(filename):
    model = th.load(filename, map_location=th.device('cpu'))
    summary(model)
    return model
