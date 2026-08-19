#!/usr/bin/env python3
"""Mixed obstacle/gate fine-tuning with gate confidence and real-flight audit."""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

ISAACSIM_REPO = Path("/home/cchen/isaacsim-workspace")
sys.path.insert(0, str(ISAACSIM_REPO))

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from gap8_perception.data import MultiTaskDataset
from gap8_perception.audit_real_flights import canonical_image_order
from gap8_perception.evaluate import local_centroid
from gap8_perception.losses import soft_dice_loss, weighted_corner_mse
from gap8_perception.model import ConvBNReLU, DSConv
from gap8_perception.temporal_data import TemporalHorizonDataset
from gap8_perception.temporal_losses import temporal_multitask_loss
from gap8_perception.train_encoder_ablation import evaluate as evaluate_obstacle
from gap8_perception.trajectory_fusion_architectures import build_trajectory_fusion_model


class NoGateDataset(Dataset):
    def __init__(self, root: Path, shard_indices):
        self.paths = []
        for index in shard_indices:
            shard = root / f"shard_{index * 1000:09d}"
            if not (shard / "_SUCCESS").is_file():
                raise FileNotFoundError(f"incomplete no-gate shard: {shard}")
            self.paths.extend(sorted(shard.glob("hm01b0_mono_*.png")))

    def __len__(self): return len(self.paths)

    def __getitem__(self, index):
        image = cv2.imread(str(self.paths[index]), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (160, 160): raise ValueError(self.paths[index])
        return {"image": torch.from_numpy(image).unsqueeze(0).float() / 255.0,
                "source": str(self.paths[index])}


class RealGateDataset(Dataset):
    def __init__(self, root: Path, flights):
        self.records = []
        for flight in flights:
            folder = root / flight
            for line in (folder / "labels.jsonl").read_text().splitlines():
                row = json.loads(line)
                corners = canonical_image_order(
                    np.asarray(row["corners"], np.float32).reshape(4, 2)
                )[0]
                if (corners < 0).any() or (corners[:, 0] >= 160).any() or (corners[:, 1] >= 160).any():
                    continue
                self.records.append((folder / "stream_out" / row["image"], corners, flight))

    def __len__(self): return len(self.records)

    def __getitem__(self, index):
        path, corners, flight = self.records[index]
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (160, 160): raise ValueError(path)
        yy, xx = np.mgrid[:40, :40]
        maps = np.zeros((4, 40, 40), np.float32)
        for channel, (x, y) in enumerate(corners / 4.0):
            maps[channel] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.25**2))
        return {"image": torch.from_numpy(image).unsqueeze(0).float() / 255.0,
                "corners": torch.from_numpy(maps), "corner_xy": torch.from_numpy(corners.copy()),
                "corner_valid": torch.tensor(True), "flight": flight, "source": str(path)}


class RetainedObstacleGateModel(nn.Module):
    def __init__(self, obstacle_checkpoint: Path, gate_checkpoint: Path, camera):
        super().__init__()
        obstacle = torch.load(obstacle_checkpoint, map_location="cpu", weights_only=False)
        self.obstacle = build_trajectory_fusion_model(
            obstacle["encoder"], obstacle["fusion"], obstacle["history"], obstacle["horizon_knots"],
            camera["camera_matrix"], camera["resolution"],
            camera.get("simulation_distortion_coefficients", camera["distortion_coefficients"]),
        )
        self.obstacle.load_state_dict(obstacle["model"], strict=True)
        gate = torch.load(gate_checkpoint, map_location="cpu", weights_only=False)
        encoder = {key.removeprefix("encoder."): value for key, value in gate["model"].items()
                   if key.startswith("encoder.")}
        self.obstacle.encoder.load_state_dict(encoder, strict=True)
        self.corner_adapter = ConvBNReLU(64, 32, 1)
        self.corner_head = nn.Sequential(DSConv(32, 16), nn.Conv2d(16, 4, 1))
        self.gate_adapter = ConvBNReLU(64, 32, 1)
        self.gate_head = nn.Sequential(DSConv(32, 16), nn.Conv2d(16, 1, 1))
        for name in ("corner_adapter", "corner_head", "gate_adapter", "gate_head"):
            prefix = name + "."
            getattr(self, name).load_state_dict({key.removeprefix(prefix): value
                for key, value in gate["model"].items() if key.startswith(prefix)}, strict=True)
        self.presence_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    @property
    def encoder(self): return self.obstacle.encoder

    def forward(self, images, frame_dt, current_motion, horizon, horizon_mask):
        return self.obstacle(images, frame_dt, current_motion, horizon, horizon_mask)

    def forward_gate(self, image):
        _, middle, _ = self.encoder(image.repeat(1, 2, 1, 1))
        corners = self.corner_head(self.corner_adapter(middle))
        mask = self.gate_head(self.gate_adapter(middle))
        return {"corners": F.interpolate(corners, (40, 40), mode="bilinear", align_corners=False),
                "gate": F.interpolate(mask, (40, 40), mode="bilinear", align_corners=False),
                "presence_logit": self.presence_head(F.adaptive_avg_pool2d(middle, 1).flatten(1)).squeeze(1)}


def move_obstacle(raw, device):
    names = ("images", "frame_dt", "current_motion", "horizon", "horizon_mask", "inverse_depth",
             "depth_valid", "flow", "flow_valid", "looming", "clearance", "collision")
    return {name: raw[name].to(device, non_blocking=True) for name in names}


def obstacle_inputs(batch):
    return {name: batch[name] for name in ("images", "frame_dt", "current_motion", "horizon", "horizon_mask")}


def gate_losses(model, synthetic, real, negative, device):
    image = synthetic["image"].to(device, non_blocking=True)
    target = synthetic["gate"].to(device, non_blocking=True)
    valid = synthetic["corner_valid"].to(device, non_blocking=True)
    output = model.forward_gate(image)
    corner = weighted_corner_mse(output["corners"], synthetic["corners"].to(device), valid)
    mask = F.binary_cross_entropy_with_logits(output["gate"], target) + 0.5 * soft_dice_loss(output["gate"], target)
    present = target.flatten(1).any(1).float()
    confidence = F.binary_cross_entropy_with_logits(output["presence_logit"], present)
    real_output = model.forward_gate(real["image"].to(device, non_blocking=True))
    real_corner = weighted_corner_mse(real_output["corners"], real["corners"].to(device),
                                       real["corner_valid"].to(device))
    real_presence = F.binary_cross_entropy_with_logits(real_output["presence_logit"],
                                                       torch.ones_like(real_output["presence_logit"]))
    no_gate = model.forward_gate(negative["image"].to(device, non_blocking=True))
    negative_mask = F.binary_cross_entropy_with_logits(no_gate["gate"], torch.zeros_like(no_gate["gate"]))
    negative_presence = F.binary_cross_entropy_with_logits(no_gate["presence_logit"],
                                                           torch.zeros_like(no_gate["presence_logit"]))
    total = 10 * corner + mask + confidence + 5 * real_corner + real_presence + negative_mask + negative_presence
    return total, {"synthetic_corner": corner, "synthetic_mask": mask, "synthetic_presence": confidence,
                   "real_corner": real_corner, "real_presence": real_presence,
                   "negative_mask": negative_mask, "negative_presence": negative_presence}


def infinite(loader):
    while True: yield from loader


def train_epoch(model, loaders, optimizer, device):
    model.train(); totals = {}; steps = max(len(loaders["obstacle"]), len(loaders["synthetic"]))
    streams = {name: infinite(loader) for name, loader in loaders.items()}
    for _ in range(steps):
        obstacle = move_obstacle(next(streams["obstacle"]), device)
        optimizer.zero_grad(set_to_none=True)
        obstacle_loss, _ = temporal_multitask_loss(model(**obstacle_inputs(obstacle)), obstacle)
        auxiliary, parts = gate_losses(model, next(streams["synthetic"]), next(streams["real"]),
                                       next(streams["negative"]), device)
        loss = obstacle_loss + auxiliary
        if not torch.isfinite(loss): raise FloatingPointError("non-finite mixed loss")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); optimizer.step()
        values = {"total": loss, "obstacle": obstacle_loss, "gate_auxiliary": auxiliary, **parts}
        for key, value in values.items(): totals[key] = totals.get(key, 0.0) + float(value.detach())
    return {key: value / steps for key, value in totals.items()}


def auc(labels, scores):
    labels=np.asarray(labels,dtype=bool); scores=np.asarray(scores); p=labels.sum(); n=(~labels).sum()
    if not p or not n: return float("nan")
    order=np.argsort(scores); ranks=np.empty_like(order,dtype=float); ranks[order]=np.arange(1,len(scores)+1)
    return float((ranks[labels].sum()-p*(p+1)/2)/(p*n))


def average_precision(labels, scores):
    labels=np.asarray(labels,dtype=bool); order=np.argsort(-np.asarray(scores)); y=labels[order]
    return float((np.cumsum(y)/np.arange(1,len(y)+1))[y].mean()) if y.any() else float("nan")


@torch.no_grad()
def evaluate_gate(model, synthetic_loader, negative_loader, real_loader, device):
    model.eval(); intersection=union=0.; errors=[]; labels=[]; scores=[]; neg_pixels=neg_total=0; neg_fp=neg_n=0
    for batch in synthetic_loader:
        output=model.forward_gate(batch["image"].to(device)); target=batch["gate"].to(device)>=.5; pred=output["gate"].sigmoid()>=.5
        intersection+=float((pred&target).sum()); union+=float((pred|target).sum())
        valid=batch["corner_valid"].to(device); xy=batch["corner_xy"].to(device); points=local_centroid(output["corners"].sigmoid())*4
        if valid.any(): errors.append(torch.linalg.vector_norm(points[valid]-xy[valid],dim=2).cpu())
        present=target.flatten(1).any(1); labels.extend(present.cpu().tolist()); scores.extend(output["presence_logit"].sigmoid().cpu().tolist())
    for batch in negative_loader:
        output=model.forward_gate(batch["image"].to(device)); pred=output["gate"].sigmoid()>=.5; probability=output["presence_logit"].sigmoid()
        neg_pixels+=int(pred.sum()); neg_total+=pred.numel(); neg_fp+=int((probability>=.5).sum()); neg_n+=len(probability)
        labels.extend([False]*len(probability)); scores.extend(probability.cpu().tolist())
    real_errors=[]; real_scores=[]
    for batch in real_loader:
        output=model.forward_gate(batch["image"].to(device)); points=local_centroid(output["corners"].sigmoid())*4
        real_errors.append(torch.linalg.vector_norm(points-batch["corner_xy"].to(device),dim=2).cpu())
        probability=output["presence_logit"].sigmoid(); real_scores.extend(probability.cpu().tolist())
    e=torch.cat(errors); re=torch.cat(real_errors); labels_np=np.asarray(labels,dtype=bool); scores_np=np.asarray(scores)
    bins=np.linspace(0,1,11); ece=0.
    for low,high in zip(bins[:-1],bins[1:]):
        chosen=(scores_np>=low)&(scores_np<(high if high<1 else high+1e-6))
        if chosen.any(): ece+=chosen.mean()*abs(scores_np[chosen].mean()-labels_np[chosen].mean())
    return {"synthetic_gate_iou":intersection/max(1.,union), "synthetic_corner_mean_px":float(e.mean()),
            "synthetic_corner_p95_px":float(torch.quantile(e,.95)), "presence_bce":float(F.binary_cross_entropy(torch.tensor(scores_np),torch.tensor(labels_np,dtype=torch.float64))),
            "presence_auroc":auc(labels_np,scores_np), "presence_ap":average_precision(labels_np,scores_np),
            "presence_accuracy":float(((scores_np>=.5)==labels_np).mean()), "presence_ece_10bin":float(ece),
            "explicit_no_gate_false_positive_rate":neg_fp/max(1,neg_n), "explicit_no_gate_mask_pixel_rate":neg_pixels/max(1,neg_total),
            "real_corner_mean_px":float(re.mean()), "real_corner_p95_px":float(torch.quantile(re,.95)),
            "real_presence_recall":float((np.asarray(real_scores)>=.5).mean()), "real_presence_mean_probability":float(np.mean(real_scores)),
            "real_examples":len(real_scores)}


def main():
    p=argparse.ArgumentParser()
    for name in ("obstacle-dataset","gate-dataset","gate-targets","gate-split-file","no-gate-dataset","real-root","obstacle-checkpoint","gate-checkpoint","output"):
        p.add_argument(f"--{name}",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=20); p.add_argument("--workers",type=int,default=8); p.add_argument("--seed",type=int,default=20260819)
    p.add_argument("--obstacle-batch-size",type=int,default=32); p.add_argument("--gate-batch-size",type=int,default=64); p.add_argument("--learning-rate",type=float,default=1e-4)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    camera=json.load(open(a.obstacle_dataset/"dataset_manifest.json"))["camera_calibration"]
    model=RetainedObstacleGateModel(a.obstacle_checkpoint,a.gate_checkpoint,camera).to(device)
    temporal={s:TemporalHorizonDataset(a.obstacle_dataset,s,2,augment=s=="train",minimum_current_index=2) for s in ("train","validation","test")}
    synthetic={s:MultiTaskDataset(a.gate_dataset,a.gate_targets,a.gate_split_file,s) for s in ("train","validation","test")}
    negative={"train":NoGateDataset(a.no_gate_dataset,(0,1,2)),"validation":NoGateDataset(a.no_gate_dataset,(3,)),"test":NoGateDataset(a.no_gate_dataset,(4,))}
    real={"train":RealGateDataset(a.real_root,("flight_06",)),"validation":RealGateDataset(a.real_root,("flight_07",)),"test":RealGateDataset(a.real_root,("flight_08",))}
    def loader(ds,batch,shuffle): return DataLoader(ds,batch,shuffle=shuffle,num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0)
    train_loaders={"obstacle":loader(temporal["train"],a.obstacle_batch_size,True),"synthetic":loader(synthetic["train"],a.gate_batch_size,True),"negative":loader(negative["train"],a.gate_batch_size,True),"real":loader(real["train"],a.gate_batch_size,True)}
    val_obstacle=loader(temporal["validation"],a.obstacle_batch_size,False); test_obstacle=loader(temporal["test"],a.obstacle_batch_size,False)
    val_gate=[loader(synthetic["validation"],a.gate_batch_size,False),loader(negative["validation"],a.gate_batch_size,False),loader(real["validation"],a.gate_batch_size,False)]
    test_gate=[loader(synthetic["test"],a.gate_batch_size,False),loader(negative["test"],a.gate_batch_size,False),loader(real["test"],a.gate_batch_size,False)]
    original_state=torch.load(a.obstacle_checkpoint,map_location="cpu",weights_only=False)["model"]
    current_state={k:v.cpu().clone() for k,v in model.obstacle.state_dict().items()}; model.obstacle.load_state_dict(original_state)
    original_validation=evaluate_obstacle(model,val_obstacle,device); original_test=evaluate_obstacle(model,test_obstacle,device)
    model.obstacle.load_state_dict(current_state); premix_validation=evaluate_obstacle(model,val_obstacle,device); premix_test=evaluate_obstacle(model,test_obstacle,device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=1e-4); best=float("inf"); history=[]
    for epoch in range(1,a.epochs+1):
        train=train_epoch(model,train_loaders,optimizer,device); obstacle_val=evaluate_obstacle(model,val_obstacle,device); gate_val=evaluate_gate(model,*val_gate,device)
        score=obstacle_val["loss"]+gate_val["synthetic_corner_mean_px"]/10+(1-gate_val["synthetic_gate_iou"])+gate_val["presence_bce"]
        record={"epoch":epoch,"train":train,"obstacle_validation":obstacle_val,"gate_validation":gate_val,"selection_score":score}; history.append(record); print(json.dumps(record),flush=True)
        state={"epoch":epoch,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"record":record}
        torch.save(state,a.output/"last.pt")
        if score<best: best=score; torch.save(state,a.output/"best.pt")
    saved=torch.load(a.output/"best.pt",map_location=device,weights_only=False); model.load_state_dict(saved["model"])
    final={"best_epoch":saved["epoch"],"original_obstacle_validation":original_validation,"original_obstacle_test":original_test,
           "gate_only_finetuned_obstacle_validation":premix_validation,"gate_only_finetuned_obstacle_test":premix_test,
           "mixed_obstacle_validation":evaluate_obstacle(model,val_obstacle,device),"mixed_obstacle_test":evaluate_obstacle(model,test_obstacle,device),
           "mixed_gate_validation":evaluate_gate(model,*val_gate,device),"mixed_gate_test":evaluate_gate(model,*test_gate,device),
           "real_split":{"train":["flight_06"],"validation":["flight_07"],"test":["flight_08"]},"history":history,
           "confidence":"sigmoid(presence_logit), trained with binary cross entropy","threshold":0.5}
    (a.output/"summary.json").write_text(json.dumps(final,indent=2)+"\n"); print(json.dumps(final),flush=True)


if __name__=="__main__": main()
