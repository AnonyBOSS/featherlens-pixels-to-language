"""EDA team batch example; preparation path adapted for this repository."""
from pathlib import Path
import sys
import random
import numpy as np
import torch
if not (Path.cwd()/"src/data/classification.py").is_file():
    raise SystemExit("Run this example from the repository root.")
sys.path.insert(0,str(Path.cwd()))
from src.data.preparation import load_classification_manifest
from src.data.source import load_cub_split
from src.data.classification import CubClassificationDataset,make_classification_loaders
from src.data.transforms import classification_transform

def main():
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    rows=load_classification_manifest(Path(__file__).resolve().parent)
    source=load_cub_split("train")
    datasets={p:CubClassificationDataset(source,rows,partition=p,
              transform=classification_transform(training=p=="train")) for p in ("train","val")}
    loaders=make_classification_loaders(datasets,batch_size=4,num_workers=0,seed=42)
    print({p:len(dataset) for p,dataset in datasets.items()})
    for partition,loader in loaders.items():
        batch=next(iter(loader))
        assert set(batch)=={"image","label","image_id"} and batch["image"].shape==(4,3,224,224)
        assert batch["image"].dtype==torch.float32 and batch["label"].dtype==torch.int64
        print(partition,tuple(batch["image"].shape),batch["image"].dtype,batch["label"].dtype)
        print(batch["image_id"][0])
if __name__=="__main__":
    main()
