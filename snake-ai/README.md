# SnakeAI

[简体中文](README_CN.md) | English | [日本語](README_JP.md)

This project contains the program scripts for the classic game "Snake" and an artificial intelligence agent that can play the game automatically. The intelligent agent is trained using deep reinforcement learning and includes two versions: an agent based on a Multi-Layer Perceptron (MLP) and an agent based on a Convolution Neural Network (CNN), with the latter having a higher average game score.

### File Structure

```bash
├── main
│   ├── logs
│   ├── sound
│   ├── trained_models_cnn
│   ├── trained_models_cnn_mps
│   ├── trained_models_mlp
│   ├── snake_game.py
│   ├── snake_env.py
│   ├── train.py
│   ├── evaluate.py
│   ├── train_cnn.py / train_mlp.py
│   └── test_cnn.py / test_mlp.py
├── utils
├── pyproject.toml
└── requirements.txt
```

The main code folder for the project is `main/`. `snake_game.py` contains the Pygame game implementation. `snake_env.py` contains the shared Gymnasium environment implementation for both CNN and MLP observations. `train.py` and `evaluate.py` are the preferred command-line entry points. The older `train_cnn.py`, `train_mlp.py`, `test_cnn.py`, and `test_mlp.py` files are kept as compatibility wrappers.

`logs/` includes TensorBoard data. `trained_models_cnn/`, `trained_models_cnn_mps/`, and `trained_models_mlp/` contain saved model weights.

The other folder `utils/` includes two utility scripts. `check_gpu_status/` is used to check if the GPU can be called by PyTorch; `compress_code.py` can remove all indentation and line breaks from the code, turning it into a tightly arranged single line of text for easier communication with GPT-4 when asking for code suggestions (GPT-4's understanding of code is far superior to humans and doesn't require indentation, line breaks, etc.).

## Running Guide

This project is based on Python and uses [Gymnasium](https://gymnasium.farama.org/), [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/), [SB3-Contrib](https://sb3-contrib.readthedocs.io/), and Pygame. The modernized code targets Python 3.10 or newer.

### Environment Configuration

```bash
# Create a conda environment named SnakeAI with Python 3.10+
conda create -n SnakeAI python=3.10
conda activate SnakeAI

# [Optional] To use a CUDA GPU for training, install the PyTorch build that matches your CUDA setup.
# See https://pytorch.org/get-started/locally/ for the current command.

# [Optional] Run the script to test if PyTorch can successfully call the GPU
python utils/check_gpu_status.py

# Install external code libraries
pip install -r requirements.txt
```

### Running Tests

The `main/` folder of the project contains the program scripts for the classic game "Snake", based on the [Pygame](https://www.pygame.org/news) code library. You can directly run the following command to play the game:

```bash
cd snake-ai/main
python snake_game.py
```

After completing the environment configuration, use `evaluate.py` to test a trained agent:

```bash
cd snake-ai/main
python evaluate.py --agent cnn
python evaluate.py --agent mlp --no-render
```

The evaluator reports average score, reward, steps per food, and short-loop hits. The reward-shaping values used during training can also be passed to evaluation:

```bash
python evaluate.py --agent mlp --no-render --food-time-penalty 0.002 --food-step-limit-multiplier 1.5 --food-reward-bonus 0.8 --distance-reward-scale 0.02 --loop-penalty 0.03 --oscillation-penalty 0.03
```

You can also keep using the compatibility scripts:

```bash
python test_cnn.py
python test_mlp.py
```

Model weight files are stored in the `main/trained_models_cnn/`, `main/trained_models_cnn_mps/`, and `main/trained_models_mlp/` folders. If older model files fail to load after the dependency upgrade, retrain with the new SB3/SB3-Contrib version or install the original legacy requirements.

### Training Models

If you need to retrain the models, use `train.py`:

```bash
cd snake-ai/main
python train.py --agent cnn --device auto
python train.py --agent mlp --device auto
```

CLI 會在載入 PyTorch 前把 BLAS/OpenMP 執行緒設為 `SNAKE_NATIVE_THREADS`（預設 1），
並預設使用一個 PyTorch CPU intra-op thread，避免小型 Snake 網路在共享主機上因
oversubscription 反而卡住；專用主機可在實測後同時調整 `SNAKE_NATIVE_THREADS=N` 與
`--torch-threads N`。

The environment includes configurable food reward, distance shaping, and penalties for slow food collection, repeated short loops, and compact oscillation patterns:

```bash
python train.py --agent mlp --food-reward-bonus 0.8 --distance-reward-scale 0.02 --food-time-penalty 0.002 --food-step-limit-multiplier 1.5 --loop-penalty 0.03 --loop-window 16 --oscillation-penalty 0.03 --oscillation-window 12
```

For a quick smoke training run:

```bash
python train.py --agent cnn --total-timesteps 4096 --num-envs 2 --no-stdout-log
```

CNN observations are always exact 84×84 `uint8` images. The board size must
divide 84 exactly (supported values: 3, 4, 6, 7, 12, 14, 21, 28, 42, and 84),
otherwise the CLI fails before creating training output. CNN training defaults
to CHW observations; `--no-cnn-channel-first` selects HWC, which SB3 transposes
internally. Both layouts use the same guarded promotion protocol. Run
`python train.py --help` to inspect these options without starting training.

`--checkpoint-interval-timesteps` is measured in total environment transitions,
independent of `--num-envs` (`--checkpoint-interval` remains a compatibility
alias). Training output is appended to `training_log.txt` with a session marker,
so continuing from `--load-model` preserves the earlier audit trail.

The CLI evaluates the pre-training and candidate policies on paired deterministic
development seeds plus a separate fixed holdout. `ppo_snake_final.zip` is written
only when both suites show a material behavior/food improvement (and an existing
protected final is also beaten). Short smoke runs below
`--guard-min-training-timesteps` still exit successfully, but write
`ppo_snake_candidate_unverified.zip` and `training_guard_report.json`; periodic
checkpoint filenames also include `candidate_unverified` and must not be treated
as verified models. A successful promotion also writes the durable evidence
sidecar `ppo_snake_final.guard.json` and embeds the same `training_guard.json`
inside the atomically promoted model archive; later rejected attempts do not
overwrite either protected artifact.

The compatibility scripts still work:

```bash
python train_cnn.py
python train_mlp.py
```

### GUI Training Preview (Unverified Candidates Only)

`./run_gui_train_demo.sh` is an interactive training preview, not an official
promotion path. It deliberately skips the paired development/fixed-holdout
guard used by `train.py`, so its Pygame title, status text, and terminal output
all label the policy as **UNVERIFIED CANDIDATE**. The demo writes only
`ppo_snake_<agent>_gui_demo_candidate_unverified.zip` plus a matching
`.guard.json` report; the bundle embeds the same report with
`verified: false`. It never writes `ppo_snake_final.zip` or a dashboard
protected-best bundle. Run `train.py` (or guarded dashboard PPO training) when
the result must be eligible for verified promotion. The report records the
actual `model.num_timesteps` delta (which may exceed the requested chunk because
SB3 completes full rollouts). Pressing Ctrl+C is treated as a normal interactive
stop: the partial candidate is saved with `termination_reason:
keyboard_interrupt`. Other runtime errors still propagate and are not disguised
as a successful demo run.

### Interactive Web Dashboard

You can watch training and adjust live parameters in a browser:

```bash
cd snake-ai
./run_web_dashboard.sh
```

Then open `http://localhost:7860/`.

The dashboard trains in guarded PPO chunks by default and renders a labelled preview on an HTML Canvas. Selecting the Hamiltonian strategy changes only the preview; it is never counted as PPO guard evidence or a best-model score. Disable **Enable real PPO weight training** only when you intentionally want a preview-only run. Model-shape settings such as CNN/MLP, device, rollout steps, and batch size should be applied with **Reset With Config**.

Whenever PPO training is enabled, the mandatory guard retains a candidate only after a
deterministic same-seed evaluation measures a real behavior improvement and a
separate fixed-seed holdout does not regress. Rejected chunks are rolled back
and remain recorded in the exported history.

Use **Download Model** to save a `.snakeai.zip` bundle containing the Stable-Baselines3 model plus dashboard metadata. Upload is disabled by default because SB3 bundles contain trusted Python serialization. For trusted local bundles, launch with `SNAKE_ENABLE_MODEL_UPLOAD=1 ./run_web_dashboard.sh`, then use **Import And Pause**. A manual import loads its weights only as an unverified baseline: imported history, old guard claims, and protected-best provenance are cleared, so fresh guarded training must establish new evidence.

At startup the dashboard resumes only an internally protected bundle with the
exact current dashboard format (v3) and fixed-holdout protocol. MLP, CNN, and
each complete CNN architecture/layout use separate agent+protocol checkpoint
namespaces. Promotion evidence must be accepted, include real attempted steps,
show development improvement and holdout non-regression, and bind the exact
embedded `model.zip` SHA-256. Before serving the weights, the dashboard loads
them and reproduces the stored fixed-holdout score/food/objective. Legacy,
future-version, malformed, replaced, or non-reproducible bundles are moved to a
quarantine filename and the dashboard falls back to the repository baseline
(or a newly initialized model). Merely possessing changed weights or editable
guard metadata does not make a model verified.

### Viewing Curves

The project includes Tensorboard curve graphs of the training process. You can use Tensorboard to view detailed data. It is recommended to use the integrated Tensorboard plugin in VSCode for direct viewing, or you can use the traditional method:

```bash
cd snake-ai/main
tensorboard --logdir=logs/
```

Open the default Tensorboard service address `http://localhost:6006/` in your browser to view the interactive curve graphs of the training process.

## Acknowledgements
The external code libraries used in this project include Gymnasium, Pygame, Stable-Baselines3, and SB3-Contrib. Thanks all the software developers for their selfless dedication to the open-source community!

The convolutional neural network used in this project is from the Nature paper:

[1] [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
