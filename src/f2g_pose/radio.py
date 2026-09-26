"""RADIO is an independently licensed, explicitly configured dependency."""
import os
from pathlib import Path
import torch


def load_frozen_radio(config):
    repository = getattr(config, 'radio_repository', None) or os.environ.get('F2G_RADIO_REPOSITORY')
    checkpoint = getattr(config, 'radio_checkpoint', None) or os.environ.get('F2G_RADIO_CHECKPOINT')
    if not repository or not Path(repository).is_dir():
        raise FileNotFoundError('Provide --radio-repository or F2G_RADIO_REPOSITORY (local RADIO checkout).')
    if not checkpoint or not Path(checkpoint).is_file():
        raise FileNotFoundError('Provide --radio-checkpoint or F2G_RADIO_CHECKPOINT (RADIO v2.5-L weights).')
    model = torch.hub.load(str(repository), 'radio_model', version=str(checkpoint),
                           source='local', skip_validation=True, progress=False)
    model.requires_grad_(False)
    model.eval()
    return model
