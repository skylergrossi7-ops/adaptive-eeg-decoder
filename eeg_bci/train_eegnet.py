"""CPU training on the supplied training partition only."""

import numpy as np
import torch
from torch import nn

from eeg_bci.eegnet import EEGNet


def as_microvolts(volts):
    return torch.from_numpy(np.ascontiguousarray(volts * 1e6, dtype=np.float32))


def fit_eegnet(volts, labels, *, seed, epochs, batch_size, learning_rate):
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("Training settings must be positive.")
    if len(volts) != len(labels) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Expected matched EEG trials with both binary hand labels.")
    eeg = as_microvolts(volts)
    targets = torch.as_tensor(labels, dtype=torch.long)
    if not torch.isfinite(eeg).all():
        raise ValueError("Nonfinite training signal.")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = EEGNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        for _ in range(epochs):
            for indices in torch.randperm(len(targets)).split(batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(eeg[indices]), targets[indices])
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss.")
                loss.backward()
                optimizer.step()
                model.constrain_weights()
    return model.eval()


@torch.inference_mode()
def predict_proba(model, volts):
    model.eval()  # Disable dropout and freeze training-fold batch-normalization statistics.
    return model(as_microvolts(volts)).softmax(dim=1).numpy()
