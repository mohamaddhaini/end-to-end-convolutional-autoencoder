# End-to-End Convolutional Autoencoder

This repository contains the PyTorch implementation of the nonlinear hyperspectral
unmixing model described in
"[End-to-end convolutional autoencoder for nonlinear hyperspectral unmixing](https://www.mdpi.com/2072-4292/14/14/3341)".
The encoder estimates endmember abundances directly from hyperspectral pixels while
the decoder reconstructs both linear and nonlinear spectral contributions.

## Repository Structure

- `conv_autoencoder.py` – model definition, training loop, abundance extraction, and preprocessing helpers.
- `endmember_number_estimation.py` – HySime-based utilities for estimating the number of endmembers (signal subspace size).
- `walk-through.ipynb` / `endmembers_number.ipynb` – example notebooks that demonstrate the workflow.

## Requirements

The code was tested on Python 3.8+ with PyTorch 1.13. Install the dependencies
listed below (or adapt to your environment/GPU build):

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy pandas matplotlib scikit-learn spectral pysptools barbar tensorboard tqdm
```

## Data Preparation

1. Convert your hyperspectral cube to a `torch.utils.data.Dataset` that returns
   tensors of shape `(1, nb_channel)` (batching is handled by `DataLoader`).
2. Normalize spectra using the same scaling that will be applied to the basis
   matrix (e.g., min-max scaling to `[0, 1]`).
3. Compute or provide an initial endmember matrix `basis` with shape
   `(nb_channel, nb_endm)` to initialize the decoder weights.

## Training

```python
from conv_autoencoder import ConvAutoencoder, train_nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvAutoencoder(nb_channel=288, basis=basis_tensor, nb_endm=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_loss, val_loss = train_nn(
    n_epochs=200,
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    valid_loader=valid_loader,
    nonlinear_weights_lamda=1e-3,
    smoothing=1e-6,
    path="runs/my_experiment",
    name="conv_ae.pt",
    device=device,
    linear=False,
)
```

The training loop logs TensorBoard summaries to `path/Tensorboard`, enforces
non-negativity on the decoder weights, and applies spectral divergence and smoothness
penalties for stable convergence.

## Estimating Endmember Counts

Use `endmember_number_estimation.py` to estimate the signal subspace size before
training:

```python
from endmember_number_estimation import est_noise, hysime

noise, noise_cov = est_noise(hsi_cube, noise_type="additive")
num_endmembers, _, _, costs = hysime(hsi_cube, noise, noise_cov)
```

The resulting `num_endmembers` guides the choice of `nb_endm` when instantiating
the autoencoder.

## Abundance Extraction and Reconstruction

After training:

```python
from conv_autoencoder import get_abundances, decode

abundances = get_abundances(model, test_loader, device, linear=False)
x_lin, x_nonlin, x_total = decode(model, abundances, device)
```

`get_abundances` stacks the predicted endmember proportions, while `decode`
reconstructs linear, nonlinear, and combined spectra.

## Citation

If you use this work, please cite:

Dhaini, Mohamad, et al. "End-to-end convolutional autoencoder for nonlinear hyperspectral unmixing." Remote Sensing 14.14 (2022): 3341.

```bibtex
@article{dhaini2022end,
  title={End-to-end convolutional autoencoder for nonlinear hyperspectral unmixing},
  author={Dhaini, Mohamad and Berar, Maxime and Honeine, Paul and Van Exem, Antonin},
  journal={Remote Sensing},
  volume={14},
  number={14},
  pages={3341},
  year={2022},
  publisher={MDPI}
}
```

## License

Please refer to the original publication for licensing terms. Contact the authors
for clarifications regarding redistribution or commercial use.
