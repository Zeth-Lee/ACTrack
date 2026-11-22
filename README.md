# ACTrack

## Installation

The environment uses PyTorch 2.5.1 and CUDA 12.1. From the repository root, install the dependencies in a new environment:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install --no-deps -e third_party/sam3
```

For DiDi evaluation, also install:

```bash
python -m pip install vot-toolkit==0.8.1
```

Use the bundled `third_party/` sources, which include the SAM3 memory selection and motion prior used by ACTrack.

## Model weights

Download the checkpoints through the [MCITrack](https://github.com/kangben258/MCITrack) and [SAM3](https://github.com/facebookresearch/sam3) instructions. The default L384 model needs:

```text
weights/
├── mcitrack_l384.pth.tar
└── sam3.pt
pretrained/
└── fast_itpn_large_1600e_1k.pt
```

The encoder initialization file under `pretrained/` is used during model construction. The SAM3 tokenizer vocabulary is included in the repository.

Model paths are set in the YAML configuration and resolve relative to the repository root. Supply the VLM API key through the `ARK_API_KEY` environment variable.

## RGB tracking

Arrange each sequence as a directory of numerically named JPEG frames and a `groundtruth.txt` file containing `x, y, width, height` annotations:

```text
data/lasot/airplane/airplane-1/
├── img/
│   ├── 00000001.jpg
│   ├── 00000002.jpg
│   └── ...
└── groundtruth.txt
```

Run a sequence from its first-frame box:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_rgb.py \
  --config configs/rgb/l384.yaml \
  --frames data/lasot/airplane/airplane-1/img \
  --init-file data/lasot/airplane/airplane-1/groundtruth.txt \
  --output outputs/airplane-1
```

The run saves `predictions.txt`, matching confidence, frame logs, and a completion record. Choose a new output directory for each run.

Evaluate the predictions:

```bash
python scripts/evaluate_rgb.py \
  --predictions outputs/airplane-1/predictions.txt \
  --groundtruth data/lasot/airplane/airplane-1/groundtruth.txt
```

For a full dataset, save a sequence manifest as `run.json` in the repository root, with one entry per sequence:

```json
{
  "sequences": [
    {
      "name": "airplane-1",
      "frames": "data/lasot/airplane/airplane-1/img",
      "init_file": "data/lasot/airplane/airplane-1/groundtruth.txt"
    }
  ]
}
```

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_benchmark.py \
  --manifest run.json --config configs/rgb/l384.yaml \
  --output outputs/lasot
```

## VLM prompt language

Choose one language in the `agent` section of the run's YAML configuration:

```yaml
agent:
  vlm_reprompt_prompt_language: zh  # default; set to en for English
```

| Setting | Prompt | Requested response labels |
| --- | --- | --- |
| `zh` | Original Chinese prompt | `Tracker`, `SAM`, `Uncertain` |
| `en` | Faithful English translation with source names mapped to the paper terminology | `Matching`, `Motion`, `Uncertain` |

Both prompts live in [prompts.py](src/actrack/agent/prompts.py). Only the selected
prompt is sent. English mode also uses Matching/Motion in the four-panel titles.
The translation retains the original confidence, similar-object, occlusion and
whole/part rules. Different languages can still produce different VLM verdicts.

The English parser accepts a single label (case-insensitive, optionally quoted)
and maps `Matching` to internal `Tracker` and `Motion` to internal `SAM`. The
original Chinese parser is preserved. Invalid English responses follow the
existing API failure path and return `Uncertain`; raw responses remain in logs.
Gate logs record `prompt_language`; `answer` always uses the internal vocabulary.

Both languages share the configured event branches, K=5 episode gate and one-time
correction reuse. A Matching verdict selects Matching and reprompts SAM3; a Motion
verdict selects Motion without reprompting; Uncertain retains the default choice.
The supplied configurations keep Matching and Perception search factors at 2.5.
This branch has CPU interface checks; no new tracking score is attributed to the
language refactor.

## VLM trigger extensions

Branch 1 (`identity_diff_motion`) checks different valid instance assignments,
at least two detections, and nonoverlapping boxes (IoU <= 1e-9). One switch controls
whether the two additional events are included:

```yaml
agent:
  vlm_reprompt_gate_enabled: true
  extend_branch: true  # false: Branch 1 only; true: all three branches
```

| `extend_branch` | Enabled trigger events |
| --- | --- |
| `false` | Branch 1 (`identity_diff_motion`) only. |
| `true` (default) | Branch 1 plus Branch 2 (`unassigned_tracker`) and Branch 3 (`same_instance_extent`). |

Branch 2 checks a missing assignment on either side, 1 or 2 detections, and
IoU <= 1e-9. Branch 3 checks the same assigned instance with the Matching center
inside the Motion box and the Motion center outside the Matching box.

The default `true` preserves the full three-event pipeline. Use YAML booleans
`true`/`false`, without quotes. The existing `vlm_reprompt_gate_enabled: false`
disables the entire VLM gate, including Branch 1.

`extend_branch` applies to both prompt languages. When it is `false`, Branch 2/3
events do not accumulate an episode, call the VLM, or consume the one-time
correction reuse. Enabled events retain the same K=5/once-per-episode rules.
The default box-selection policy, verdict actions, and search factors are
unchanged. Gate logs record the setting as `vlm_reprompt_gate.extend_branch`.

## DiDi evaluation

Place the dataset under `data/DiDi`, with `list.txt` and one directory per sequence. Each sequence contains `color/` frames and `groundtruth.txt`.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_didi.py \
  --dataset data/DiDi --workspace workspaces/didi_full \
  --score
```

The adapter initializes from the first-frame region and returns bounding boxes with matching confidence. Each sequence runs in a fresh process. Use `--sequence VOT2022-ST_leaves` for one sequence or `--prepare-only` to create the workspace.

## Acknowledgements

We thank [MCITrack](https://github.com/kangben258/MCITrack) and [SAM3](https://github.com/facebookresearch/sam3) for their implementations and pretrained models.
