# Legacy reference code

This folder contains the unmodified original ICCV-2021 HiNet reference implementation
(`train.py`, `test.py`, `config.py`, `hinet.py`, `model.py`, `invblock.py`,
`rrdb_denselayer.py`, `datasets.py`, `viz.py`, `util.py`, `train_logging.py`,
`calculate_PSNR_SSIM.py`, and `modules/`).

It is kept here for reference only. The active training and evaluation entry
points have been refactored into the `src/` package and live at the repo root:

- `main.py` — training driver (multi-stage, gradient safety, optional noise layer)
- `evaluate.py` — post-training evaluation with attack-robustness suite
- `src/{core,data,engine,models}/` — refactored modules

To run anything in this folder, `cd legacy/` first so the flat `import config`,
`from model import *`, and `import modules.Unet_common` style still resolves.
Paths inside `config.py` will need to be adjusted accordingly.
