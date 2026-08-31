from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from verl.utils.dataset.rl_dataset import RLHFDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = _load_script("prepare_vero35k", PROJECT_ROOT / "scripts" / "prepare_vero35k.py")


def test_metadata_download_never_allows_image_archives(tmp_path, monkeypatch):
    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(prepare, "snapshot_download", fake_snapshot_download)
    prepare.download_metadata("example/repo", tmp_path)
    assert captured["repo_type"] == "dataset"
    assert set(captured["allow_patterns"]) == set(prepare.METADATA_FILES)
    assert all("images.tar" not in pattern for pattern in captured["allow_patterns"])


def test_build_training_jsonl_preserves_verl_schema_and_resolves_images(tmp_path):
    image_a = tmp_path / "images" / "a.png"
    image_b = tmp_path / "images" / "b.png"
    image_a.parent.mkdir()
    image_a.write_bytes(b"a")
    image_b.write_bytes(b"b")
    source = tmp_path / prepare.SOURCE_JSONL
    item = {
        "id": "sample-1",
        "data_source": "fixture",
        "prompt": [{"role": "user", "content": "<image><image>\nquestion"}],
        "images": ["images/a.png", "images/b.png"],
        "ability": "fixture",
        "reward_model": {"style": "rule", "ground_truth": "answer"},
        "extra_info": {"answer": "answer"},
        "vision_selector_budget": [0.2, 0.1],
    }
    source.write_text(json.dumps(item) + "\n", encoding="utf-8")
    output = tmp_path / prepare.OUTPUT_JSONL

    assert prepare.build_training_jsonl(source, output, tmp_path) == 1
    converted = json.loads(output.read_text(encoding="utf-8"))
    assert converted["prompt"] == item["prompt"]
    assert converted["reward_model"] == item["reward_model"]
    assert converted["extra_info"] == item["extra_info"]
    assert converted["vision_selector_budget"] == item["vision_selector_budget"]
    assert converted["images"] == [{"path": str(image_a.resolve())}, {"path": str(image_b.resolve())}]


def test_build_training_jsonl_rejects_missing_image(tmp_path):
    source = tmp_path / prepare.SOURCE_JSONL
    source.write_text(
        json.dumps({"prompt": [{"role": "user", "content": "<image>"}], "images": ["images/missing.png"]})
        + "\n",
        encoding="utf-8",
    )
    try:
        prepare.build_training_jsonl(source, tmp_path / prepare.OUTPUT_JSONL, tmp_path)
    except FileNotFoundError as exc:
        assert "missing.png" in str(exc)
    else:
        raise AssertionError("missing image was accepted")


def test_build_training_jsonl_can_limit_real_data_smoke_subset(tmp_path):
    image = tmp_path / "images" / "sample.png"
    image.parent.mkdir()
    image.write_bytes(b"image")
    source = tmp_path / prepare.SOURCE_JSONL
    records = [
        {"id": index, "prompt": [{"role": "user", "content": "<image>"}], "images": ["images/sample.png"]}
        for index in range(2)
    ]
    source.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    output = tmp_path / prepare.OUTPUT_JSONL

    assert prepare.build_training_jsonl(source, output, tmp_path, limit=1) == 1
    assert len(output.read_text(encoding="utf-8").splitlines()) == 1


def test_verl_dataset_accepts_jsonl_files(tmp_path, monkeypatch):
    jsonl_path = tmp_path / "train.verl.jsonl"
    jsonl_path.write_text("{}\n", encoding="utf-8")
    loaded = [{"prompt": []}]

    def fake_load_dataset(format_name, *, data_files):
        assert format_name == "json"
        assert data_files == str(jsonl_path)
        return {"train": loaded}

    import verl.utils.dataset.rl_dataset as rl_dataset_module

    monkeypatch.setattr(rl_dataset_module.datasets, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(rl_dataset_module.datasets, "concatenate_datasets", lambda datasets: datasets[0])
    dataset = RLHFDataset.__new__(RLHFDataset)
    dataset.data_files = [str(jsonl_path)]
    dataset.max_samples = -1
    dataset.shuffle = False
    dataset.seed = None
    dataset.maybe_filter_out_long_prompts = lambda dataframe: dataframe
    dataset._read_files_and_tokenize()
    assert dataset.dataframe == loaded
