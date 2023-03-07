import os
from datetime import datetime

import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import GuitarSet, MAESTRO
from evaluate import evaluate
from constants import *
from autoregressive.models import AR_Transcriber
from onsets_and_frames.transcriber import OnsetsAndFrames
from utils import summary, cycle

ex = Experiment('train_transcriber')


def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    iterations = 100000
    resume_iteration = None
    checkpoint_interval = 6000

    train_on = 'GuitarSet'
    train_with = 'ar'
    logdir = 'runs/transcriber-' + train_with + '-' + datetime.now().strftime('%y%m%d-%H%M%S')

    batch_size = 4
    sequence_length = 327680 // 2
    model_complexity = 48

    if torch.cuda.is_available() and torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory < 10e9:
        batch_size //= 2
        sequence_length //= 2
        print(f'Reducing batch size to {batch_size} and sequence_length to {sequence_length} to save memory')

    learning_rate = 0.0006
    learning_rate_decay_steps = 2000
    learning_rate_decay_rate = 0.98

    leave_one_out = None

    clip_gradient_norm = 3

    validation_length = None
    validation_interval = 2000

    ex.observers.append(FileStorageObserver(logdir))

    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    train_groups, validation_groups = ['train'], ['validation']

    if leave_one_out is not None:
        all_years = {'2004', '2006', '2008', '2009', '2011', '2013', '2014', '2015', '2017'}
        train_groups = list(all_years - {str(leave_one_out)})
        validation_groups = [str(leave_one_out)]

    if train_on == 'MAESTRO':
        dataset = MAESTRO(groups=train_groups, sequence_length=sequence_length)
        validation_dataset = MAESTRO(groups=validation_groups, sequence_length=sequence_length)
    else:
        dataset = GuitarSet(groups=train_groups, sequence_length=sequence_length)
        validation_dataset = GuitarSet(groups=validation_groups, sequence_length=validation_length)

    loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)

    if resume_iteration is None:
        model = AR_Transcriber(N_MELS, MAX_MIDI - MIN_MIDI + 1, model_complexity).to(device) if train_with == "ar" \
           else OnsetsAndFrames(N_MELS, MAX_MIDI - MIN_MIDI + 1, model_complexity).to(device)

        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
    else:
        model_path = os.path.join(logdir, f'model-{resume_iteration}.pt')
        model = torch.load(model_path)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        optimizer.load_state_dict(torch.load(os.path.join(logdir, 'last-optimizer-state.pt')))

    summary(model)
    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)

    loop = tqdm(range(resume_iteration + 1, iterations + 1))
    for i, batch in zip(loop, cycle(loader)):
        predictions, losses = model.run_on_batch(batch, train=True)

        loss = sum(losses.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if clip_gradient_norm:
            clip_grad_norm_(model.parameters(), clip_gradient_norm)

        for key, value in {'loss': loss, **losses}.items():
            writer.add_scalar(key, value.item(), global_step=i)

        if i % validation_interval == 0:
            model.eval()
            with torch.no_grad():
                scores = evaluate(validation_dataset, model)
                print("Note: %s, Offsets: %s, Frame: %s" %
                      (np.mean(scores['metric/note/f1']),
                       np.mean(scores['metric/note-with-offsets/f1']),
                       np.mean(scores['metric/frame/f1'])))
                for key, value in scores.items():
                    writer.add_scalar('validation/' + key.replace(' ', '_'), np.mean(value), global_step=i)
            model.train()

        if i % checkpoint_interval == 0:
            torch.save(model, os.path.join(logdir, f'model-{i}.pt'))
            torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))


if __name__ == "__main__":
    main()
