# PyTorch Implementation of Onsets and Frames

This is a [PyTorch](https://pytorch.org/) implementation of Google's [Onsets and Frames](https://magenta.tensorflow.org/onsets-frames) model and adaptation of the [Real-time Automatic Piano Transcription System](https://github.com/jdasam/online_amt), using the [GuitarSet dataset](https://zenodo.org/record/3371780).

## Instructions

This project is quite resource-intensive; 32 GB or larger system memory and 8 GB or larger GPU memory is recommended. 


### Downloading Dataset

Download the [audio_mono-pickup_mix](https://zenodo.org/record/3371780/files/audio_mono-pickup_mix.zip?download=1) and the [annotation](https://zenodo.org/record/3371780/files/annotation.zip?download=1) archives. Extract the annotation archive into "data/GuitarSet" and extract the audio archive into "data/GuitarSet/audio".

Install requirements and prepare the dataset with:
```bash
pip install -r requirements.txt
python guitarset_to_midi
```
This program mixes mono and comp audios to augment the dataset and converts the annotations into midi format for training.
The data is splitted in two, training split and validation split. The validation split is all the recordings of one of the 6 guitarists.


### Training

Inside train.py, you can set the model and training parameters you want to use. Model parameters are editable in their respective directories (autoregressive and onsets_and_frames).
Launch training with:
```bash
python train.py
```

The program trains and evaluate the model on the train and validation splits. Data augmentation is performed when loading the data by pitchshifting it. Pitchshift range is defined inside `constants.py`.

### Real-time transcription

To transcribe in real-time with the pianoroll animation:

```bash
python realtime_ar/run_on_plt.py --model-file "path/to/model.pt"
```

