import torch


SAMPLE_RATE = 16000
HOP_LENGTH = SAMPLE_RATE * 8 // 1000
ONSET_LENGTH = SAMPLE_RATE * 8 // 1000
OFFSET_LENGTH = SAMPLE_RATE * 8 // 1000
HOPS_IN_ONSET = ONSET_LENGTH // HOP_LENGTH
HOPS_IN_OFFSET = OFFSET_LENGTH // HOP_LENGTH
MIN_MIDI = 40
MAX_MIDI = 81

N_STATE = 4  # change for 2cnn
N_MELS = 229
MEL_FMIN = 30
MEL_FMAX = SAMPLE_RATE // 2
FILTER_LENGTH = 512

DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
