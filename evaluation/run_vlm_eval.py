"""
VLM Spatial-Reasoning Evaluation — Experiment Runner

Runs a VLM on dataset pairs and saves raw predictions.
See evaluation/README.md for full usage instructions.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── VLM backend ───────────────────────────────────────────────────────────────

SCREEN_WIDTH  = 512
SCREEN_HEIGHT = 512
MAX_RETRIES = 5              # retry attempts on timeout / 5xx errors
RETRY_BASE_WAIT = 10.0       # seconds; doubles each attempt (10, 20, 40, 80, 160)

_GOOGLE_MODEL_MAP = {
    "gemma":           "gemma-4-26b-a4b-it",
    "gemini":          "gemini-2.5-flash",
    "gemini-robotics": "gemini-robotics-er-1.6-preview",
}

QWEN_DEFAULT_MODEL_DIR = "/path/to/Qwen3-VL-8B-Instruct"


def _call_vlm(
    vlm: str,
    images: list[bytes],          # list of PNG bytes, in order
    image_labels: list[str],       # label shown before each image (e.g. "Image 0:")
    prompt_text: str,
    client_state: dict,            # mutable dict for rate-limit tracking
) -> str:
    """Send images + text to the selected VLM and return the raw text response."""
    vlm = vlm.lower()

    if vlm in _GOOGLE_MODEL_MAP:
        from google import genai
        from google.genai import types as genai_types

        worker_id = client_state.get("worker_id", 0)
        api_key = (
            os.environ.get(f"GEMINI_API_KEY_{worker_id}")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        if not api_key:
            raise RuntimeError(f"No API key found (tried GEMINI_API_KEY_{worker_id}, GEMINI_API_KEY, GOOGLE_API_KEY)")

        if "client" not in client_state:
            client_state["client"] = genai.Client(
                api_key=api_key,
                http_options={"timeout": 240_000},  # 240s (unit: milliseconds)
            )
        client = client_state["client"]
        model = _GOOGLE_MODEL_MAP[vlm]

        # rate-limit bookkeeping (30 RPM / 15K TPM for free tier)
        RPM_LIMIT = 30
        TPM_LIMIT = 15_000
        WINDOW    = 60.0
        req_times: collections.deque = client_state.setdefault("req_times", collections.deque())
        tok_log:   collections.deque = client_state.setdefault("tok_log",   collections.deque())

        def _purge(now: float) -> None:
            while req_times and now - req_times[0] >= WINDOW:
                req_times.popleft()
            while tok_log and now - tok_log[0][0] >= WINDOW:
                tok_log.popleft()

        while True:
            now = time.time()
            _purge(now)
            tokens_used = sum(t for _, t in tok_log)
            if len(req_times) < RPM_LIMIT and tokens_used < TPM_LIMIT * 0.90:
                break
            waits = []
            if len(req_times) >= RPM_LIMIT and req_times:
                waits.append(WINDOW - (now - req_times[0]))
            if tokens_used >= TPM_LIMIT * 0.90 and tok_log:
                waits.append(WINDOW - (now - tok_log[0][0]))
            wait = max(waits) + 0.5 if waits else 2.0
            print(f"  [VLM] Rate limit — waiting {wait:.1f}s...")
            time.sleep(wait)

        contents: list = []
        for label, img_bytes in zip(image_labels, images):
            contents.append(genai_types.Part.from_text(text=label))
            contents.append(genai_types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
        contents.append(genai_types.Part.from_text(text=prompt_text))

        for attempt in range(MAX_RETRIES):
            try:
                response = client.models.generate_content(model=model, contents=contents)
                ts = time.time()
                req_times.append(ts)
                if response.usage_metadata:
                    total = response.usage_metadata.total_token_count or 0
                    tok_log.append((ts, total))
                    client_state["last_tokens"] = total
                    print(f"    Tokens: {total} | window total: {sum(t for _, t in tok_log)}")
                return response.text
            except Exception as exc:
                err = str(exc)
                retryable = any(k in err for k in ("504", "DEADLINE_EXCEEDED", "timed out", "timeout", "503", "500"))
                if retryable and attempt < MAX_RETRIES - 1:
                    wait = RETRY_BASE_WAIT * (2 ** attempt)
                    print(f"    [retry {attempt + 1}/{MAX_RETRIES - 1}] {exc} — waiting {wait:.0f}s...")
                    time.sleep(wait)
                else:
                    raise

    elif vlm == "claude":
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        if "client" not in client_state:
            client_state["client"] = anthropic.Anthropic(api_key=api_key)
        client = client_state["client"]

        import base64
        content: list = []
        for label, img_bytes in zip(image_labels, images):
            content.append({"type": "text", "text": label})
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(img_bytes).decode(),
                },
            })
        content.append({"type": "text", "text": prompt_text})

        message = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=256,
            messages=[{"role": "user", "content": content}],
        )
        return message.content[0].text

    elif vlm == "gpt":
        import base64
        import openai

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        if "client" not in client_state:
            client_state["client"] = openai.OpenAI(api_key=api_key)
        client = client_state["client"]

        content: list = []
        for label, img_bytes in zip(image_labels, images):
            content.append({"type": "text", "text": label})
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64.b64encode(img_bytes).decode()}",
                },
            })
        content.append({"type": "text", "text": prompt_text})

        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model="gpt-5.4",
                    max_completion_tokens=256,
                    messages=[{"role": "user", "content": content}],
                )
                return response.choices[0].message.content
            except Exception as exc:
                err = str(exc)
                retryable = any(k in err for k in ("timeout", "timed out", "502", "503", "504", "529"))
                if retryable and attempt < MAX_RETRIES - 1:
                    wait = RETRY_BASE_WAIT * (2 ** attempt)
                    print(f"    [retry {attempt + 1}/{MAX_RETRIES - 1}] {exc} — waiting {wait:.0f}s...")
                    time.sleep(wait)
                else:
                    raise

    elif vlm == "qwen":
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        from PIL import Image
        import torch
        import io

        if "model" not in client_state:
            model_dir = client_state.get("model_dir", QWEN_DEFAULT_MODEL_DIR)
            print(f"  [Qwen] Loading model from {model_dir} ...")
            client_state["processor"] = AutoProcessor.from_pretrained(model_dir)
            client_state["model"] = Qwen3VLForConditionalGeneration.from_pretrained(
                model_dir,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            client_state["model"].eval()
            print("  [Qwen] Model loaded.")

        processor = client_state["processor"]
        model_obj = client_state["model"]

        content = []
        for label, img_bytes in zip(image_labels, images):
            content.append({"type": "text", "text": label})
            content.append({"type": "image", "image": Image.open(io.BytesIO(img_bytes)).convert("RGB")})
        content.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": content}]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model_obj.device)

        with torch.no_grad():
            output_ids = model_obj.generate(**inputs, max_new_tokens=client_state.get("max_new_tokens", 256))

        input_len = inputs["input_ids"].shape[1]
        new_tokens = [out[input_len:] for out in output_ids]
        return processor.batch_decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    else:
        raise ValueError(f"Unknown VLM backend: {vlm!r}. Choose from: gemma, gemini, gemini-robotics, claude, gpt, qwen")


# ── prompt builders ───────────────────────────────────────────────────────────

def _coord_description(width: int, height: int) -> str:
    return (
        f"Image coordinate system:\n"
        f"  - Origin [x=0, y=0] is the TOP-LEFT corner of the image.\n"
        f"  - x-axis points RIGHT: x=0 is the left edge, x={width - 1} is the right edge.\n"
        f"  - y-axis points DOWN:  y=0 is the top edge,  y={height - 1} is the bottom edge.\n"
        f"  - Image size: {width} x {height} pixels.\n"
        f"  - Valid range: x ∈ [0, {width - 1}], y ∈ [0, {height - 1}]."
    )


def _prompt_ab(sentence: str, n_images: int, width: int = SCREEN_WIDTH, height: int = SCREEN_HEIGHT) -> str:
    return (
        f"You are the intelligent brain system of a home-assistance robot. "
        f"You are given first-person perspective images captured during the robot's exploration of a home.\n\n"
        f"The {n_images} images above (labeled Image 0 through Image {n_images - 1}) were all taken "
        f"during exploration of the same single-story house. "
        f"Each image is preceded by its label (\"Image 0:\", \"Image 1:\", etc.).\n\n"
        f"Sentence: \"{sentence}\"\n\n"
        f"The resident has asked the robot to retrieve an object and is describing where they last saw it."
        f"Your mission is to predict, as accurately as possible, where that object is located, based on the resident's description in the sentence above.\n\n"
        f"Task:\n"
        f"1. Select the image (0 to {n_images - 1}) that best represents the scene described in the sentence "
        f"— the image where both the observer's furniture and the reference landmark are most clearly visible.\n"
        f"2. In the selected image, predict the center of the target object as pixel coordinates.\n"
        f"   The target object may not actually be visible in the image. "
        f"Based on the sentence, predict as faithfully and accurately as possible "
        f"where the center of the object WOULD be located.\n\n"
        f"{_coord_description(width, height)}\n\n"
        f"IMPORTANT: When predicting coordinates, pay close attention to the origin position and "
        f"axis directions, and ensure all values fall within the valid range.\n\n"
        f"Output ONLY valid JSON, no other text:\n"
        f'{{\"selected_image\": <integer 0-{n_images - 1}>, \"x\": <integer 0-{width - 1}>, \"y\": <integer 0-{height - 1}>}}'
    )

def _prompt_c(sentence: str, width: int = SCREEN_WIDTH, height: int = SCREEN_HEIGHT) -> str:
    return (
        f"You are the intelligent brain system of a home-assistance robot.\n\n"
        f"Sentence: \"{sentence}\"\n\n"
        f"The resident has asked the robot to retrieve an object and is describing where they last saw it. "
        f"Your mission is to predict, as accurately as possible, where that object is located — "
        f"based on the resident's description in the sentence above.\n\n"
        f"The image provided shows the target scene from the perspective of the resident "
        f"at the moment they spotted the target object.\n\n"
        f"Task: Based on the sentence, predict the center of the target object as pixel coordinates.\n"
        f"The target object may not actually be visible in the image. "
        f"Based on the sentence, predict as faithfully and accurately as possible "
        f"where the center of the object WOULD be located.\n\n"
        f"{_coord_description(width, height)}\n\n"
        f"IMPORTANT: When predicting coordinates, pay close attention to the origin position and "
        f"axis directions, and ensure all values fall within the valid range.\n\n"
        f"Output ONLY valid JSON, no other text:\n"
        f'{{\"x\": <integer 0-{width - 1}>, \"y\": <integer 0-{height - 1}>}}'
    )


# ── response parsing ──────────────────────────────────────────────────────────

def _extract_all_jsons(text: str) -> list[dict]:
    """Extract all JSON objects from text, in order of appearance."""
    results = []
    for m in re.finditer(r'\{[^{}]*\}', text, re.DOTALL):
        try:
            results.append(json.loads(m.group()))
        except (ValueError, TypeError):
            pass
    return results


def _best_coord_json(
    candidates: list[dict],
    keys: tuple[str, ...],
    width: int = SCREEN_WIDTH,
    height: int = SCREEN_HEIGHT,
) -> dict | None:
    """
    Pick the best JSON from candidates (last-first priority).
    Preference order:
      1. Last JSON that has all required keys AND coords within [0, w-1] x [0, h-1]
      2. Last JSON that has all required keys (regardless of range)
    """
    valid_in_range = None
    valid_any = None
    for obj in reversed(candidates):
        if not all(k in obj for k in keys):
            continue
        try:
            x = float(obj["x"])
            y = float(obj["y"])
        except (KeyError, ValueError, TypeError):
            continue
        if valid_any is None:
            valid_any = obj
        if 0 <= x <= width - 1 and 0 <= y <= height - 1:
            valid_in_range = obj
            break  # last in-range is best
    return valid_in_range if valid_in_range is not None else valid_any


def _parse_ab_response(text: str, width: int = SCREEN_WIDTH, height: int = SCREEN_HEIGHT) -> dict | None:
    """Parse {"selected_image": int, "x": float, "y": float} from VLM output.
    Uses the last in-range JSON to capture self-corrections."""
    candidates = _extract_all_jsons(text.strip())
    obj = _best_coord_json(candidates, ("selected_image", "x", "y"), width, height)
    if obj is None:
        return None
    try:
        return {
            "selected_image": int(obj["selected_image"]),
            "x": float(obj["x"]),
            "y": float(obj["y"]),
        }
    except (KeyError, ValueError, TypeError):
        return None


def _parse_c_response(text: str, width: int = SCREEN_WIDTH, height: int = SCREEN_HEIGHT) -> dict | None:
    """Parse {"x": float, "y": float} from VLM output.
    Uses the last in-range JSON to capture self-corrections."""
    candidates = _extract_all_jsons(text.strip())
    obj = _best_coord_json(candidates, ("x", "y"), width, height)
    if obj is None:
        return None
    try:
        return {"x": float(obj["x"]), "y": float(obj["y"])}
    except (KeyError, ValueError, TypeError):
        return None


# ── dataset helpers ───────────────────────────────────────────────────────────

def _load_dataset(dataset_dir: Path) -> list[dict]:
    """Load all house JSON entries, checking root then artifacts/ subdirectories."""
    house_files = sorted(dataset_dir.glob("train_house_*.json"))
    if not house_files:
        house_files = sorted((dataset_dir / "artifacts").glob("*/train_house_*.json"))

    entries: list[dict] = []
    for json_path in house_files:
        try:
            house_entries = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(house_entries, list):
                for e in house_entries:
                    e.setdefault("house_id", json_path.stem)
                entries.extend(house_entries)
        except Exception as exc:
            print(f"  Warning: failed to load {json_path.name}: {exc}")
    return entries


def _load_per_image_visible(artifacts_dir: Path, house_id: str) -> dict[str, list[str]]:
    p = artifacts_dir / house_id / "per_image_visible_objects.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _load_exploration_log(artifacts_dir: Path, house_id: str) -> dict[str, dict]:
    """Returns a mapping image_rel_path → log_entry."""
    p = artifacts_dir / house_id / "exploration_log.json"
    if not p.exists():
        return {}
    log = json.loads(p.read_text(encoding="utf-8"))
    return {entry["image_path"]: entry for entry in log if "image_path" in entry}


def _candidate_images_for_pair(
    entry: dict,
    per_image_visible: dict[str, list[str]],
) -> list[str]:
    """
    Return rel paths of exploration images where both the observer objectType
    and anchor objectType are visible.  Falls back to all images if none match.
    """
    # Get types from first non-empty command
    observer_type: str | None = None
    anchor_type:   str | None = None
    for cmd in entry.get("commands", {}).values():
        for stype in ("type_a", "type_b", "type_c"):
            c = cmd.get(stype, {})
            if c.get("sentence"):
                # derive anchor from dataset structure
                break

    # Directly read objectType from entry object_observer / object_landmark
    obs_meta = entry.get("object_observer", {})
    lm_meta  = entry.get("object_landmark", {})
    obs_raw  = obs_meta.get("raw_metadata", {})
    lm_raw   = lm_meta.get("raw_metadata", {})
    observer_type = obs_raw.get("objectType") or obs_meta.get("assetId", "").split("_")[0]
    anchor_type   = lm_raw.get("objectType")  or lm_meta.get("assetId", "").split("_")[0]

    matching = [
        rel
        for rel, types in per_image_visible.items()
        if observer_type in types and anchor_type in types
    ]
    if matching:
        return sorted(matching)
    # fallback: images where at least the anchor is visible
    fallback = [rel for rel, types in per_image_visible.items() if anchor_type in types]
    return sorted(fallback) if fallback else sorted(per_image_visible.keys())


# ── main ─────────────────────────────────────────────────────────────────────

def merge_predictions(output_dir: Path, num_workers: int) -> None:
    """Merge worker files into per-house prediction JSON files (same structure as dataset)."""
    all_preds: list[dict] = []
    seen_keys: set[tuple] = set()

    def _add(preds: list[dict]) -> None:
        for p in preds:
            # house_id is part of the key — pair_id is only unique within a house
            k = (p.get("house_id", ""), p["pair_id"], p["direction"], p["type"])
            if k not in seen_keys:
                seen_keys.add(k)
                all_preds.append(p)

    # Include pre-existing single-process predictions.json
    base = output_dir / "predictions.json"
    if base.exists():
        data = json.loads(base.read_text(encoding="utf-8"))
        preds = data.get("predictions", [])
        _add(preds)
        print(f"  Loaded {len(preds)} predictions from predictions.json")

    for wid in range(num_workers):
        p = output_dir / f"predictions_{wid}.json"
        if not p.exists():
            print(f"  Warning: {p.name} not found, skipping")
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        preds = data.get("predictions", [])
        _add(preds)
        print(f"  Loaded {len(preds)} predictions from {p.name}")

    # Write one JSON file per house (mirrors dataset directory structure)
    by_house: dict[str, list[dict]] = {}
    for pred in all_preds:
        by_house.setdefault(pred.get("house_id", "unknown"), []).append(pred)

    for house_id, preds in sorted(by_house.items()):
        house_path = output_dir / f"{house_id}.json"
        house_path.write_text(
            json.dumps(preds, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  {house_id}.json: {len(preds)} predictions")

    print(f"Merged {len(all_preds)} predictions into {len(by_house)} house files → {output_dir}")


def run_eval(
    dataset_dir: Path,
    output_dir: Path,
    vlm: str = "gemma",
    types: list[str] | None = None,
    max_pairs: int | None = None,
    worker_id: int = 0,
    num_workers: int = 1,
    skip_entries: int = 0,
    qwen_model_dir: str | None = None,
) -> None:
    if types is None:
        types = ["a", "b", "c"]
    types_set = set(t.lower() for t in types)

    artifacts_dir = dataset_dir / "artifacts"
    all_entries = _load_dataset(dataset_dir)
    print(f"Loaded {len(all_entries)} dataset pairs from {dataset_dir}")

    # Apply entry offset before round-robin assignment
    if skip_entries > 0:
        all_entries = all_entries[skip_entries:]
        print(f"Skipping first {skip_entries} entries — {len(all_entries)} remaining")

    # Round-robin: each worker takes every num_workers-th entry
    entries = [e for i, e in enumerate(all_entries) if i % num_workers == worker_id]
    if num_workers > 1:
        print(f"Worker {worker_id}/{num_workers}: assigned {len(entries)} entries")

    output_dir.mkdir(parents=True, exist_ok=True)
    # Each worker writes its own file to avoid conflicts
    fname = "predictions.json" if num_workers == 1 else f"predictions_{worker_id}.json"
    output_path = output_dir / fname

    # Resume: load existing predictions and skip already-done items
    predictions: list[dict] = []
    done_keys: set[tuple] = set()
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            predictions = existing.get("predictions", [])
            done_keys = {(p.get("house_id", ""), p["pair_id"], p["direction"], p["type"]) for p in predictions}
            print(f"Resuming: {len(predictions)} predictions already saved, skipping those.")
        except Exception as exc:
            print(f"  Warning: could not load existing predictions: {exc}")

    # In worker mode, also skip pairs already done in the single-process predictions.json
    if num_workers > 1:
        base_pred = output_dir / "predictions.json"
        if base_pred.exists():
            try:
                base_data = json.loads(base_pred.read_text(encoding="utf-8"))
                base_keys = {(p.get("house_id", ""), p["pair_id"], p["direction"], p["type"]) for p in base_data.get("predictions", [])}
                done_keys |= base_keys
                print(f"  Also skipping {len(base_keys)} pairs already in predictions.json")
            except Exception as exc:
                print(f"  Warning: could not load predictions.json for dedup: {exc}")

    client_state: dict = {
        "worker_id": worker_id,
        "model_dir": qwen_model_dir or QWEN_DEFAULT_MODEL_DIR,
    }

    run_start = time.time()
    cmd_count = 0    # number of VLM calls made this session (excluding skipped/resumed)
    total_tokens = 0  # cumulative token count across all VLM calls

    log_path = output_dir / f"timing_{worker_id}.log"

    def _log(msg: str) -> None:
        print(msg)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    metadata_base = {
        "dataset_dir":  str(dataset_dir),
        "vlm":          vlm,
        "types":        sorted(types_set),
        "screen_width":  SCREEN_WIDTH,
        "screen_height": SCREEN_HEIGHT,
        "worker_id":    worker_id,
        "num_workers":  num_workers,
    }

    def _flush() -> None:
        output_path.write_text(
            json.dumps(
                {
                    **metadata_base,
                    "timestamp_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "n_predictions": len(predictions),
                    "predictions":   predictions,
                },
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    processed = 0
    for entry in entries:
        if max_pairs is not None and processed >= max_pairs:
            break

        pair_id  = entry.get("pair_id", "?")
        house_id = entry.get("house_id", "")

        # New dataset format (dataset_generation_new3): sentences are in stage5/stage6,
        # camera and rgb_path are at the top level.
        stage5 = entry.get("stage5", {})
        stage6 = entry.get("stage6", {})
        rgb_rel = entry.get("rgb_path")
        camera  = entry.get("camera", {})

        per_image_visible = _load_per_image_visible(artifacts_dir, house_id)
        house_artifacts   = artifacts_dir / house_id

        # Pass all exploration images of the house (excluding depth maps)
        exp_dir = house_artifacts / "exploration_images"
        candidate_rel_paths = sorted([
            f"exploration_images/{p.name}"
            for p in exp_dir.glob("*.png")
            if "depth" not in p.name
        ]) if exp_dir.exists() else _candidate_images_for_pair(entry, per_image_visible)

        # Collect all directions that have any content
        all_directions = sorted(set(stage5.keys()) | set(stage6.keys()))

        for direction in all_directions:
            s5 = stage5.get(direction, {})
            s6 = stage6.get(direction, {})

            # ── Type A ──────────────────────────────────────────────────────
            if "a" in types_set:
                sentence = s5.get("type_a", "")
                if sentence and (house_id, pair_id, direction, "a") not in done_keys:
                    candidate_paths = [house_artifacts / rel for rel in candidate_rel_paths]
                    candidate_paths = [p for p in candidate_paths if p.exists()]
                    if candidate_paths:
                        print(f"  [{pair_id}/{direction}/A] {len(candidate_paths)} images")
                        images_bytes = [p.read_bytes() for p in candidate_paths]
                        labels = [f"Image {i}:" for i in range(len(candidate_paths))]
                        prompt = _prompt_ab(sentence, len(candidate_paths))
                        raw = ""
                        parse_error = None
                        parsed = None
                        try:
                            raw = _call_vlm(vlm, images_bytes, labels, prompt, client_state)
                            print(f"    Response: {raw[:120]}")
                            parsed = _parse_ab_response(raw)
                            if parsed is None:
                                parse_error = "failed to parse JSON"
                        except Exception as exc:
                            parse_error = str(exc)
                            print(f"    Error: {exc}")

                        tokens = client_state.pop("last_tokens", 0)
                        total_tokens += tokens
                        predictions.append({
                            "pair_id":          pair_id,
                            "house_id":         house_id,
                            "direction":        direction,
                            "type":             "a",
                            "sentence":         sentence,
                            "candidate_images": [str(p.relative_to(house_artifacts)) for p in candidate_paths],
                            "predicted_image":  candidate_rel_paths[parsed["selected_image"]]
                                                if parsed and 0 <= parsed["selected_image"] < len(candidate_rel_paths)
                                                else None,
                            "predicted_x":      parsed["x"]  if parsed else None,
                            "predicted_y":      parsed["y"]  if parsed else None,
                            "raw_response":     raw,
                            "parse_error":      parse_error,
                            "tokens":           tokens,
                        })
                        _flush()
                        processed += 1
                        cmd_count += 1

            # ── Type B ──────────────────────────────────────────────────────
            if "b" in types_set:
                sentence = s5.get("type_b", "")
                if sentence and (house_id, pair_id, direction, "b") not in done_keys:
                    candidate_paths = [house_artifacts / rel for rel in candidate_rel_paths]
                    candidate_paths = [p for p in candidate_paths if p.exists()]
                    if candidate_paths:
                        print(f"  [{pair_id}/{direction}/B] {len(candidate_paths)} images")
                        images_bytes = [p.read_bytes() for p in candidate_paths]
                        labels = [f"Image {i}:" for i in range(len(candidate_paths))]
                        prompt = _prompt_ab(sentence, len(candidate_paths))
                        raw = ""
                        parse_error = None
                        parsed = None
                        try:
                            raw = _call_vlm(vlm, images_bytes, labels, prompt, client_state)
                            print(f"    Response: {raw[:120]}")
                            parsed = _parse_ab_response(raw)
                            if parsed is None:
                                parse_error = "failed to parse JSON"
                        except Exception as exc:
                            parse_error = str(exc)
                            print(f"    Error: {exc}")

                        tokens = client_state.pop("last_tokens", 0)
                        total_tokens += tokens
                        predictions.append({
                            "pair_id":          pair_id,
                            "house_id":         house_id,
                            "direction":        direction,
                            "type":             "b",
                            "sentence":         sentence,
                            "candidate_images": [str(p.relative_to(house_artifacts)) for p in candidate_paths],
                            "predicted_image":  candidate_rel_paths[parsed["selected_image"]]
                                                if parsed and 0 <= parsed["selected_image"] < len(candidate_rel_paths)
                                                else None,
                            "predicted_x":      parsed["x"]  if parsed else None,
                            "predicted_y":      parsed["y"]  if parsed else None,
                            "raw_response":     raw,
                            "parse_error":      parse_error,
                            "tokens":           tokens,
                        })
                        _flush()
                        processed += 1
                        cmd_count += 1

            # ── Type C ──────────────────────────────────────────────────────
            if "c" in types_set:
                sentence = s6.get("type_c", "")
                if sentence and (house_id, pair_id, direction, "c") not in done_keys:
                    rgb_path = house_artifacts / rgb_rel if rgb_rel else None
                    if rgb_path and rgb_path.exists():
                        print(f"  [{pair_id}/{direction}/C]")
                        img_bytes = rgb_path.read_bytes()
                        c_w = int(camera.get("screen_width",  SCREEN_WIDTH))
                        c_h = int(camera.get("screen_height", SCREEN_HEIGHT))
                        prompt = _prompt_c(sentence, c_w, c_h)
                        raw = ""
                        parse_error = None
                        parsed = None
                        try:
                            raw = _call_vlm(vlm, [img_bytes], ["Image:"], prompt, client_state)
                            print(f"    Response: {raw[:120]}")
                            parsed = _parse_c_response(raw)
                            if parsed is None:
                                parse_error = "failed to parse JSON"
                        except Exception as exc:
                            parse_error = str(exc)
                            print(f"    Error: {exc}")

                        tokens = client_state.pop("last_tokens", 0)
                        total_tokens += tokens
                        predictions.append({
                            "pair_id":       pair_id,
                            "house_id":      house_id,
                            "direction":     direction,
                            "type":          "c",
                            "sentence":      sentence,
                            "gt_image":      rgb_rel,
                            "predicted_x":   parsed["x"] if parsed else None,
                            "predicted_y":   parsed["y"] if parsed else None,
                            "raw_response":  raw,
                            "parse_error":   parse_error,
                            "tokens":        tokens,
                        })
                        _flush()
                        processed += 1
                        cmd_count += 1

    _flush()

    elapsed = time.time() - run_start
    mins, secs = divmod(int(elapsed), 60)
    rate = cmd_count / elapsed * 60 if elapsed > 0 else 0.0
    summary = (
        f"\n── Worker {worker_id} summary ──────────────────────────\n"
        f"  Commands executed : {cmd_count}\n"
        f"  Elapsed time      : {mins}m {secs:02d}s\n"
        f"  Throughput        : {rate:.1f} commands/min\n"
        f"  Total tokens      : {total_tokens:,}\n"
        f"  Output            : {output_path}\n"
        f"────────────────────────────────────────────────────"
    )
    _log(summary)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # Load .env from evaluation/ dir (two levels up from this script) if present
    try:
        from dotenv import load_dotenv
        _env_path = Path(__file__).resolve().parents[1] / ".env"
        if _env_path.exists():
            load_dotenv(dotenv_path=_env_path)
            print(f"Loaded env: {_env_path}")
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Run VLM evaluation on a dataset.")
    parser.add_argument("--dataset-dir", required=True, help="Path to dataset directory (contains *.json and artifacts/).")
    parser.add_argument("--output-dir",  required=True, help="Directory to save predictions.json.")
    parser.add_argument("--vlm", default="gemini",
                        choices=["gemma", "gemini", "gemini-robotics", "claude", "gpt", "qwen"],
                        help="VLM backend (default: gemini).")
    parser.add_argument("--qwen-model-dir", default=None,
                        help=f"Local path to Qwen3-VL model directory (default: {QWEN_DEFAULT_MODEL_DIR}).")
    parser.add_argument("--types", nargs="+", default=["a", "b", "c"], choices=["a", "b", "c"],
                        help="Sentence types to evaluate (default: a b c).")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Stop after this many predictions (for quick tests).")
    parser.add_argument("--worker-id",   type=int, default=0,
                        help="This worker's index (0-based). Default: 0.")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Total number of parallel workers. Default: 1 (no split).")
    parser.add_argument("--merge", action="store_true",
                        help="Merge predictions_{0..N-1}.json into predictions.json and exit.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.merge:
        merge_predictions(output_dir, args.num_workers)
        return

    run_eval(
        dataset_dir=Path(args.dataset_dir),
        output_dir=output_dir,
        vlm=args.vlm,
        types=args.types,
        max_pairs=args.max_pairs,
        worker_id=args.worker_id,
        num_workers=args.num_workers,
        qwen_model_dir=args.qwen_model_dir,
    )


if __name__ == "__main__":
    main()
