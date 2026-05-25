# Dataset

The benchmark dataset is available for download from Google Drive:

**[Download (Google Drive)] (https://drive.google.com/file/d/1U8VIv5i5vzp58BT5qZXK6XDnSfFkknze/view?usp=sharing)**

After downloading, extract the archive so the directory looks like:

```
dataset/
└── 0519_house54/
    ├── train_house_00000.json
    ├── train_house_00001.json
    ├── ...
    └── artifacts/
        ├── train_house_00000/
        │   ├── exploration_images/
        │   └── observers/
        ├── train_house_00001/
        └── ...
```

## Dataset structure

Each `train_house_XXXXX.json` contains a list of evaluation pairs:

```json
[
  {
    "pair_id": "pair_000001",
    "stage5": {
      "right": { "type_a": "...", "type_b": "..." },
      "left":  { "type_a": "...", "type_b": "..." },
      ...
    },
    "stage6": {
      "right": { "type_c": "..." },
      ...
    },
    "rgb_path": "observers/pair_000001.png",
    "camera":   { "screen_width": 512, "screen_height": 512 },
    "gt_x": 0.312,
    "gt_y": 0.578
  }
]
```

- **Type A** — observer activity described without naming the observer's furniture.
- **Type B** — same as A, but the observer's furniture is explicitly named.
- **Type C** — factual spatial statement only; paired with a ground-truth observer image.

`artifacts/<house_id>/exploration_images/` contains up to 30 RGB exploration frames (512×512 PNG) and corresponding depth visualisations.

`artifacts/<house_id>/observers/` contains the ground-truth observer-perspective image for each pair (512×512 PNG), used for Type C evaluation.
