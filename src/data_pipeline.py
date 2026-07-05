"""Data loading, label cleaning, splits, and torchvision transforms."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from src.utils import PROJECT_ROOT, class_to_idx, ensure_dir, resolve_path, set_seed


class DefectImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: Path,
        image_col: str,
        label_col: str,
        class_map: dict[str, int],
        transform=None,
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.image_col = image_col
        self.label_col = label_col
        self.class_map = class_map
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = self.image_dir / row[self.image_col]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.class_map[row[self.label_col]]
        return image, label


def unpack_zip(cfg: dict) -> dict:
    """Extract data/ from zip; skip __MACOSX."""
    zip_path = resolve_path(cfg["data"]["zip_path"])
    if not zip_path.exists():
        alt = PROJECT_ROOT / "data.zip"
        if alt.exists():
            zip_path = alt
        else:
            raise FileNotFoundError(f"Zip topilmadi: {zip_path}")

    raw_dir = resolve_path(cfg["data"]["raw_images_dir"])
    label_csv = resolve_path(cfg["data"]["label_csv"])
    ensure_dir(raw_dir.parent)

    extracted_images = 0
    extracted_csv = False

    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.startswith("__MACOSX") or name.endswith("/"):
                continue
            if not name.startswith("data/"):
                continue
            rel = name[len("data/") :]
            if not rel:
                continue
            target = raw_dir.parent / rel
            if name.endswith("/"):
                ensure_dir(target)
                continue
            ensure_dir(target.parent)
            with zf.open(name) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            if rel.startswith("raw_images/"):
                extracted_images += 1
            if rel == "label.csv":
                extracted_csv = True

    return {
        "zip_path": str(zip_path),
        "images": extracted_images,
        "label_csv": str(label_csv),
        "csv_found": extracted_csv,
    }


def load_and_clean_labels(cfg: dict) -> pd.DataFrame:
    label_path = resolve_path(cfg["data"]["label_csv"])
    df = pd.read_csv(label_path)
    img_col = cfg["data"]["image_column"]
    lbl_col = cfg["data"]["label_column"]
    classes = cfg["data"]["classes"]

    df = df.dropna(subset=[img_col, lbl_col])
    df[lbl_col] = df[lbl_col].astype(str).str.strip().str.lower()
    df = df[df[lbl_col].isin(classes)].copy()

    policy = cfg["data"].get("duplicate_policy", "drop_conflicting")
    dup_mask = df.duplicated(subset=[img_col], keep=False)
    if dup_mask.any():
        if policy == "drop_conflicting":
            grouped = df.groupby(img_col)[lbl_col].nunique()
            conflict = grouped[grouped > 1].index
            df = df[~df[img_col].isin(conflict)].copy()
        elif policy == "keep_first":
            df = df.drop_duplicates(subset=[img_col], keep="first")
        else:
            df = df.drop_duplicates(subset=[img_col], keep="last")

    return df.reset_index(drop=True)


def create_splits(df: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    seed = cfg.get("random_seed", 42)
    lbl_col = cfg["data"]["label_column"]
    train_r = cfg["data"]["train_ratio"]
    val_r = cfg["data"]["val_ratio"]
    test_r = cfg["data"]["test_ratio"]

    if abs(train_r + val_r + test_r - 1.0) > 1e-6:
        raise ValueError("train/val/test nisbati jami 1 bo'lishi kerak.")

    train_df, temp_df = train_test_split(
        df, test_size=(1 - train_r), stratify=df[lbl_col], random_state=seed
    )
    relative_val = val_r / (val_r + test_r)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - relative_val), stratify=temp_df[lbl_col], random_state=seed
    )
    return {"train": train_df, "val": val_df, "test": test_df}


def save_splits(splits: dict[str, pd.DataFrame], cfg: dict) -> Path:
    processed = resolve_path(cfg["data"]["processed_dir"])
    ensure_dir(processed)
    for name, split_df in splits.items():
        split_df.to_csv(processed / f"{name}.csv", index=False)
    meta = {
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
        "classes": cfg["data"]["classes"],
    }
    (processed / "split_info.json").write_text(
        __import__("json").dumps(meta, indent=2), encoding="utf-8"
    )
    return processed


def build_transforms(cfg: dict, train: bool = True) -> transforms.Compose:
    size = cfg["data"]["image_size"]
    aug = cfg.get("augmentation", {})
    t_list = []

    if train and aug.get("random_resized_crop"):
        t_list.append(transforms.RandomResizedCrop(size, scale=(0.85, 1.0)))
    else:
        t_list.append(transforms.Resize((size, size)))

    if train:
        if aug.get("horizontal_flip"):
            t_list.append(transforms.RandomHorizontalFlip())
        rot = aug.get("rotation_degrees", 12)
        if rot:
            t_list.append(transforms.RandomRotation(rot))
        cj = aug.get("color_jitter")
        if cj:
            t_list.append(
                transforms.ColorJitter(
                    brightness=cj.get("brightness", 0.1),
                    contrast=cj.get("contrast", 0.1),
                    saturation=cj.get("saturation", 0.1),
                )
            )

    t_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(t_list)


def load_split_dataframe(split: str, cfg: dict) -> pd.DataFrame:
    path = resolve_path(cfg["data"]["processed_dir"]) / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split topilmadi: {path}. Avval prepare_data.py ishga tushiring.")
    return pd.read_csv(path)


def make_dataloader(
    split: str,
    cfg: dict,
    batch_size: int | None = None,
    train: bool | None = None,
    shuffle: bool | None = None,
) -> DataLoader:
    if train is None:
        train = split == "train"
    df = load_split_dataframe(split, cfg)
    raw_dir = resolve_path(cfg["data"]["raw_images_dir"])
    cmap = class_to_idx(cfg["data"]["classes"])
    transform = build_transforms(cfg, train=train)
    ds = DefectImageDataset(
        df, raw_dir, cfg["data"]["image_column"], cfg["data"]["label_column"], cmap, transform
    )
    bs = batch_size or cfg["training"]["batch_size"]
    nw = cfg["training"]["num_workers"]
    if shuffle is None:
        shuffle = train
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=nw, pin_memory=True)


def prepare_dataset(cfg: dict) -> dict:
    set_seed(cfg.get("random_seed", 42))
    df = load_and_clean_labels(cfg)
    splits = create_splits(df, cfg)
    out = save_splits(splits, cfg)
    return {"processed_dir": str(out), "total": len(df), **{k: len(v) for k, v in splits.items()}}
