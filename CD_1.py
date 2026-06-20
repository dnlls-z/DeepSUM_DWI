from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.io import loadmat

def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_config_value(
    cfg: dict[str, Any],
    key: str,
    default: Any = None,
    *,
    required: bool = True,
) -> Any:
    if key in cfg:
        return cfg[key]

    for section in ("hyperparameters", "others", "paths", "dataset", "training"):
        section_data = cfg.get(section)
        if isinstance(section_data, dict) and key in section_data:
            return section_data[key]

    if required:
        raise KeyError(f"Missing required config value: {key}")

    return default


def require_path(value: str | Path, key: str) -> Path:
    path = Path(value)
    if not str(path):
        raise ValueError(f"Config value '{key}' is empty.")
    return path


@dataclass(frozen=True)
class CreateDatasetConfig:
    config_path: Path
    library_dir: Path
    mat_path: Path
    dataset_path: Path
    testset_path: Path

    nickname: str
    band: str

    scale: int
    n_chunks: int
    patch_size: int
    patches_per_set: int
    smart_patching: bool

    slices: int
    num_subj: int
    train_subj: int
    expected_reps: int
    registration_ref: int
    shift_abs_threshold: float

    @property
    def imgs_per_subject(self) -> int:
        if self.num_subj <= 0:
            raise ValueError("num_subj must be > 0.")
        if self.slices % self.num_subj != 0:
            raise ValueError(
                f"slices={self.slices} is not divisible by num_subj={self.num_subj}."
            )
        return self.slices // self.num_subj

    @property
    def train_end(self) -> int:
        return self.imgs_per_subject * self.train_subj


def load_create_dataset_config(config_path: str | Path) -> CreateDatasetConfig:
    config_path = Path(config_path).expanduser().resolve()
    cfg = read_json(config_path)

    default_library_dir = (
        config_path.parents[2].parent / "BN_global_files"
        if len(config_path.parents) >= 3
        else Path.cwd()
    )

    patch_size_hr = get_config_value(cfg, "patch_size_HR", [96, 96], required=False)
    if isinstance(patch_size_hr, (list, tuple)):
        patch_size = int(patch_size_hr[0])
    else:
        patch_size = int(patch_size_hr)

    return CreateDatasetConfig(
        config_path=config_path,
        library_dir=Path(
            get_config_value(cfg, "library_dir", default_library_dir, required=False)
        ).expanduser().resolve(),
        mat_path=require_path(get_config_value(cfg, "mat_path"), "mat_path"),
        dataset_path=require_path(get_config_value(cfg, "dataset_path"), "dataset_path"),
        testset_path=require_path(get_config_value(cfg, "testset_path"), "testset_path"),
        nickname=str(get_config_value(cfg, "nickname")),
        band=str(get_config_value(cfg, "spectral_band", "NIR", required=False)),
        scale=int(get_config_value(cfg, "scale", get_config_value(cfg, "R", 3, required=False), required=False)),
        n_chunks=int(get_config_value(cfg, "n_chunks")),
        patch_size=patch_size,
        patches_per_set=int(get_config_value(cfg, "num_patches_per_set", 20, required=False)),
        smart_patching=bool(get_config_value(cfg, "smart_patching", True, required=False)),
        slices=int(get_config_value(cfg, "slices")),
        num_subj=int(get_config_value(cfg, "num_subj")),
        train_subj=int(get_config_value(cfg, "train_subj")),
        expected_reps=int(get_config_value(cfg, "numband", get_config_value(cfg, "T_in", 9, required=False), required=False)),
        registration_ref=int(get_config_value(cfg, "registration_ref", 1, required=False)),
        shift_abs_threshold=float(get_config_value(cfg, "shift_abs_threshold", 4, required=False)),
    )


def import_project_functions(library_dir: Path) -> None:
    library_dir = Path(library_dir).expanduser().resolve()
    if str(library_dir) not in sys.path:
        sys.path.insert(0, str(library_dir))


@dataclass
class SplitData:
    lr: np.ndarray | list[np.ndarray]
    hr: np.ndarray
    mlr: np.ndarray | list[np.ndarray]
    mhr: np.ndarray
    norm: np.ndarray | None = None
    shifts: list[np.ndarray] | None = None


def lr_volume_to_imagesets(volume: np.ndarray, dtype=np.uint16) -> np.ndarray:
    return np.transpose(volume, (2, 3, 0, 1)).astype(dtype)


def hr_volume_to_imagesets(volume: np.ndarray) -> np.ndarray:
    return np.transpose(volume, (2, 0, 1))


def load_dwi_data(mat_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mat_path = Path(mat_path).expanduser().resolve()
    if not mat_path.exists():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

    mat = loadmat(mat_path)

    required = ["dwi_lr", "dwi_hr", "mask_lr", "mask_hr", "norm_value"]
    missing = [key for key in required if key not in mat]
    if missing:
        raise KeyError(f"Missing keys in MAT file: {missing}")

    lr = lr_volume_to_imagesets(mat["dwi_lr"])
    hr = hr_volume_to_imagesets(mat["dwi_hr"])
    mlr = lr_volume_to_imagesets(mat["mask_lr"])
    mhr = hr_volume_to_imagesets(mat["mask_hr"])
    norm = np.asarray(mat["norm_value"].T)

    return lr, hr, mlr, mhr, norm


def split_dataset(
    lr: np.ndarray,
    hr: np.ndarray,
    mlr: np.ndarray,
    mhr: np.ndarray,
    norm: np.ndarray,
    config: CreateDatasetConfig,
) -> tuple[SplitData, SplitData, SplitData]:
    train_end = config.train_end

    train = SplitData(
        lr=lr[:train_end],
        hr=hr[:train_end],
        mlr=mlr[:train_end],
        mhr=mhr[:train_end],
        norm=norm[:train_end],
    )

    testval = SplitData(
        lr=lr[train_end:],
        hr=hr[train_end:],
        mlr=mlr[train_end:],
        mhr=mhr[train_end:],
        norm=norm[train_end:],
    )

    val_size = len(testval.lr) // 2

    valid = SplitData(
        lr=testval.lr[:val_size],
        hr=testval.hr[:val_size],
        mlr=testval.mlr[:val_size],
        mhr=testval.mhr[:val_size],
        norm=testval.norm[:val_size],
    )

    test = SplitData(
        lr=testval.lr[val_size:],
        hr=testval.hr[val_size:],
        mlr=testval.mlr[val_size:],
        mhr=testval.mhr[val_size:],
        norm=testval.norm[val_size:],
    )

    return train, valid, test


def group_bad_repetitions(
    shifts: Iterable[np.ndarray],
    threshold: float,
) -> dict[int, list[int]]:
    bad = defaultdict(list)

    for imageset_idx, imageset_shifts in enumerate(shifts):
        for rep_idx, shift in enumerate(imageset_shifts):
            if (np.abs(shift) > threshold).any():
                bad[imageset_idx].append(rep_idx)

    return dict(bad)


def remove_bad_repetitions(
    lr_sets: list[np.ndarray],
    mlr_sets: list[np.ndarray],
    shifts: list[np.ndarray],
    bad_by_set: dict[int, list[int]],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    for idx, bad_reps in bad_by_set.items():
        lr_sets[idx] = np.delete(lr_sets[idx], bad_reps, axis=0)
        mlr_sets[idx] = np.delete(mlr_sets[idx], bad_reps, axis=0)
        shifts[idx] = np.delete(shifts[idx], bad_reps, axis=0)

    return lr_sets, mlr_sets, shifts


def drop_incomplete_imagesets(data: SplitData, expected_reps: int, label: str) -> SplitData:
    incomplete = [i for i, x in enumerate(data.lr) if np.asarray(x).shape[0] < expected_reps]

    if not incomplete:
        print(f"[{label}] removed sets (<{expected_reps} imgs): []")
        return data

    keep = [i for i in range(len(data.lr)) if i not in set(incomplete)]

    data.lr = [data.lr[i] for i in keep]
    data.mlr = [data.mlr[i] for i in keep]
    data.hr = np.delete(data.hr, incomplete, axis=0)
    data.mhr = np.delete(data.mhr, incomplete, axis=0)

    if data.norm is not None:
        data.norm = np.delete(data.norm, incomplete, axis=0)

    if data.shifts is not None:
        data.shifts = [data.shifts[i] for i in keep]

    print(f"[{label}] removed sets (<{expected_reps} imgs): {incomplete}")
    return data


def upsample_register_and_clean(data: SplitData, config: CreateDatasetConfig, label: str) -> SplitData:
    from UDN_1 import (
        registration_imageset_against_best_image_without_union_mask,
        upsampling_mask_all_imageset,
        upsampling_without_aggregation_all_imageset,
    )

    lr = np.asarray([np.asarray(x) for x in data.lr])
    mlr = np.asarray([np.asarray(x) for x in data.mlr])

    lr_up = upsampling_without_aggregation_all_imageset(lr, scale=config.scale)
    mlr_up = upsampling_mask_all_imageset(mlr, scale=config.scale)

    _, _, shifts, new_index_orders = registration_imageset_against_best_image_without_union_mask(
        lr_up,
        mlr_up,
        config.registration_ref,
    )

    lr_up = np.asarray([np.asarray(x) for x in lr_up])
    mlr_up = np.asarray([np.asarray(x) for x in mlr_up])

    lr_up = [imageset[new_index_orders[i]] for i, imageset in enumerate(lr_up)]
    mlr_up = [imageset[new_index_orders[i]] for i, imageset in enumerate(mlr_up)]
    shifts = [np.asarray(x) for x in shifts]

    bad_by_set = group_bad_repetitions(shifts, threshold=config.shift_abs_threshold)
    print(f"[{label}] removed img indices per set: {bad_by_set}")

    lr_up, mlr_up, shifts = remove_bad_repetitions(lr_up, mlr_up, shifts, bad_by_set)

    cleaned = SplitData(
        lr=lr_up,
        hr=data.hr,
        mlr=mlr_up,
        mhr=data.mhr,
        norm=data.norm,
        shifts=shifts,
    )
    cleaned = drop_incomplete_imagesets(cleaned, expected_reps=config.expected_reps, label=label)

    print(f"[{label}] final LR sets: {len(cleaned.lr)}")
    print(f"[{label}] final HR shape: {cleaned.hr.shape}")

    return cleaned



def save_numpy_map(save_map: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, array in save_map.items():
        np.save(out_dir / filename, array, allow_pickle=True)


def save_training_patches(train: SplitData, config: CreateDatasetConfig) -> None:
    from UDN_1 import create_patch_dataset_return_shifts

    for chunk_idx in range(config.n_chunks):
        patch_data = create_patch_dataset_return_shifts(
            train.lr,
            train.hr,
            train.mlr,
            train.mhr,
            train.shifts,
            patch_size=config.patch_size,
            num_patches_per_set=config.patches_per_set,
            scale=1,
            smart_patching=config.smart_patching,
        )

        save_map = {
            f"{chunk_idx}_dataset_{config.band}_patch_{config.nickname}_s_LR.npy": patch_data["training_patch"],
            f"{chunk_idx}_dataset_{config.band}_patch_{config.nickname}_s_HR.npy": patch_data["training_y_patch"],
            f"{chunk_idx}_dataset_{config.band}_patch_mask_{config.nickname}_s_LR.npy": patch_data["training_mask_patch"],
            f"{chunk_idx}_dataset_{config.band}_patch_mask_{config.nickname}_s_HR.npy": patch_data["training_mask_y_patch"],
            f"{chunk_idx}_shifts_patch_{config.nickname}_s_{config.band}.npy": patch_data["shifts"],
            f"{chunk_idx}_coordinates_{config.nickname}_s_{config.band}.npy": patch_data["coordinates"],
        }

        save_numpy_map(save_map, config.dataset_path)


def save_validation(valid: SplitData, config: CreateDatasetConfig) -> None:
    save_map = {
        f"dataset_{config.band}_{config.nickname}_s_LR_valid.npy": valid.lr,
        f"dataset_{config.band}_{config.nickname}_s_HR_valid.npy": valid.hr,
        f"dataset_{config.band}_mask_{config.nickname}_s_LR_valid.npy": valid.mlr,
        f"dataset_{config.band}_mask_{config.nickname}_s_HR_valid.npy": valid.mhr,
        f"shifts_valid_{config.nickname}_s_{config.band}.npy": valid.shifts,
        f"norm_{config.nickname}_s_{config.band}.npy": valid.norm,
    }
    save_numpy_map(save_map, config.dataset_path)


def save_test(test: SplitData, config: CreateDatasetConfig) -> None:
    save_map = {
        f"dataset_{config.band}_{config.nickname}_s_LR_test.npy": test.lr,
        f"dataset_{config.band}_{config.nickname}_s_HR_test.npy": test.hr,
        f"dataset_{config.band}_mask_{config.nickname}_s_LR_test.npy": test.mlr,
        f"dataset_{config.band}_mask_{config.nickname}_s_HR_test.npy": test.mhr,
        f"shifts_test_{config.nickname}_s_{config.band}.npy": test.shifts,
        f"norm_{config.nickname}_{config.band}_test.npy": test.norm,
    }
    save_numpy_map(save_map, config.testset_path)



def mean_std_from_npy(files: list[str]) -> tuple[float, float, int]:
    total_n = 0
    total_sum = 0.0
    total_sq_sum = 0.0

    for fp in files:
        arr = np.asarray(np.load(fp, allow_pickle=True), dtype=np.float64)
        total_n += arr.size
        total_sum += float(arr.sum())
        total_sq_sum += float(np.square(arr).sum())

    if total_n == 0:
        raise ValueError("No pixels found in files.")

    mean = total_sum / total_n
    var = max((total_sq_sum / total_n) - mean**2, 0.0)
    return float(mean), float(np.sqrt(var)), int(total_n)


def update_deepsum_config(config: CreateDatasetConfig) -> None:
    lr_files = sorted(
        glob.glob(str(config.dataset_path / f"*_dataset_{config.band}_patch_{config.nickname}_s_LR.npy"))
    )
    hr_files = sorted(
        glob.glob(str(config.dataset_path / f"*_dataset_{config.band}_patch_{config.nickname}_s_HR.npy"))
    )

    print(f"Found LR patch files: {len(lr_files)}")
    print(f"Found HR patch files: {len(hr_files)}")

    if not lr_files or not hr_files:
        raise FileNotFoundError(
            f"Could not find patch files in {config.dataset_path}. Expected patterns: "
            f"*_dataset_{config.band}_patch_{config.nickname}_s_LR.npy and "
            f"*_dataset_{config.band}_patch_{config.nickname}_s_HR.npy"
        )

    mu_lr, sigma_lr, n_lr = mean_std_from_npy(lr_files)
    _, sigma_hr, n_hr = mean_std_from_npy(hr_files)

    cfg = read_json(config.config_path)

    if "others" not in cfg:
        cfg["others"] = {}

    old_values = {k: cfg["others"].get(k) for k in ("mu", "sigma", "sigma_rescaled")}

    new_values = {
        "mu": mu_lr,
        "sigma": sigma_lr,
        "sigma_rescaled": sigma_hr,
    }
    cfg["others"].update(new_values)

    with config.config_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4)

    print("\nConfig updated successfully.")
    print("Config path:", config.config_path)
    print("Old values:", old_values)
    print("New values:", new_values)
    print(f"Pixels used -> LR: {n_lr}, HR: {n_hr}")


def print_config_summary(config: CreateDatasetConfig) -> None:
    print("Nuevo nickname:", config.nickname)
    print("dataset path:", config.dataset_path)
    print("testset path:", config.testset_path)
    print("nickname:", config.nickname)
    print("spectral band:", config.band)
    print("mat_path:", config.mat_path)
    print()
    print("slices:", config.slices)
    print("num_subj:", config.num_subj)
    print("train_subj:", config.train_subj)
    print("imgs_per_subject:", config.imgs_per_subject)
    print("registration_ref:", config.registration_ref)
    print("shift_abs_threshold:", config.shift_abs_threshold)
    print("scale:", config.scale)
    print()
    print("n_chunks:", config.n_chunks)
    print("patch_size:", config.patch_size)
    print("num_patches_per_set:", config.patches_per_set)
    print("smart_patching:", config.smart_patching)


def main(config_path: str | Path) -> CreateDatasetConfig:
    config = load_create_dataset_config(config_path)
    import_project_functions(config.library_dir)

    import UDN_1  

    config.dataset_path.mkdir(parents=True, exist_ok=True)
    config.testset_path.mkdir(parents=True, exist_ok=True)

    print_config_summary(config)
    print("Carpeta Dataset:", config.dataset_path.resolve())
    print("Carpeta Testset:", config.testset_path.resolve())

    lr, hr, mlr, mhr, norm = load_dwi_data(config.mat_path)

    print(
        f"LR: {lr.shape} HR: {hr.shape} MLR: {mlr.shape} "
        f"MHR: {mhr.shape} NORM: {norm.shape}"
    )

    train, valid, test = split_dataset(lr, hr, mlr, mhr, norm, config)

    print(f"Train: {len(train.lr)} Val: {len(valid.lr)} Test: {len(test.lr)}")

    train = upsample_register_and_clean(train, config, "TRAIN")
    valid = upsample_register_and_clean(valid, config, "VAL")
    test = upsample_register_and_clean(test, config, "TEST")

    save_training_patches(train, config)
    save_validation(valid, config)
    save_test(test, config)

    update_deepsum_config(config)

    print("Create_Dataset se ejecutó con éxito")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create DeepSUM dataset from config.")
    parser.add_argument(
        "--config",
        "--config_path",
        dest="config_path",
        required=True,
        help="Path to config JSON file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(config_path=args.config_path)
