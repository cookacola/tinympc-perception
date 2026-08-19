#!/usr/bin/env python3
"""Train gate-opening heads tapped at different encoder depths."""
from __future__ import annotations
import argparse, csv, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from gap8_perception.data import MultiTaskDataset
from gap8_perception.losses import soft_dice_loss
from gap8_perception.model import ConvBNReLU, DSConv, Gap8MultiTaskNet

class GateTapNet(nn.Module):
    """Same encoder and decoder, with a minimal tap-specific adapter."""
    def __init__(self, tap: str):
        super().__init__()
        if tap not in {"early80", "mid40", "late40"}: raise ValueError(tap)
        base = Gap8MultiTaskNet()
        self.stem, self.e1_down, self.e1_refine = base.stem, base.e1_down, base.e1_refine
        self.geometry40, self.tap = base.geometry40, tap
        self.adapter = {"early80": DSConv(12, 16, 2), "mid40": ConvBNReLU(20, 16, 1),
                        "late40": nn.Identity()}[tap]
        self.head = nn.Sequential(DSConv(16, 12), nn.Conv2d(12, 1, 1))
    def forward(self, image):
        early = self.stem(image)
        if self.tap == "early80": feature = early
        else:
            middle = self.e1_refine(self.e1_down(early))
            feature = middle if self.tap == "mid40" else self.geometry40(middle)
        return self.head(self.adapter(feature))

def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None); totals = {k: 0.0 for k in ("loss","bce","dice","intersection","union","predicted","truth")}; samples = 0
    with (torch.enable_grad() if optimizer is not None else torch.no_grad()):
        for batch in loader:
            image, target = batch["image"].to(device, non_blocking=True), batch["gate"].to(device, non_blocking=True)
            if optimizer is not None: optimizer.zero_grad(set_to_none=True)
            logits = model(image); bce = F.binary_cross_entropy_with_logits(logits, target); dice = soft_dice_loss(logits, target); loss = bce + 0.5*dice
            if optimizer is not None: loss.backward(); optimizer.step()
            prediction, truth, size = logits.sigmoid() >= 0.5, target >= 0.5, image.shape[0]
            totals["loss"] += float(loss)*size; totals["bce"] += float(bce)*size; totals["dice"] += float(dice)*size
            totals["intersection"] += float((prediction & truth).sum()); totals["union"] += float((prediction | truth).sum())
            totals["predicted"] += float(prediction.sum()); totals["truth"] += float(truth.sum()); samples += size
    inter = totals["intersection"]
    return {"loss": totals["loss"]/max(1,samples), "bce": totals["bce"]/max(1,samples), "soft_dice_loss": totals["dice"]/max(1,samples),
            "iou": inter/max(1.0,totals["union"]), "f1": 2*inter/max(1.0,totals["predicted"]+totals["truth"]),
            "precision": inter/max(1.0,totals["predicted"]), "recall": inter/max(1.0,totals["truth"])}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--tap",choices=("early80","mid40","late40"),required=True)
    for name in ("dataset","targets","split-file","output"): p.add_argument(f"--{name}",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=60); p.add_argument("--batch-size",type=int,default=128); p.add_argument("--workers",type=int,default=8)
    p.add_argument("--seed",type=int,default=20260819); p.add_argument("--learning-rate",type=float,default=2e-3); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train=MultiTaskDataset(a.dataset,a.targets,a.split_file,"train",augment=True); val=MultiTaskDataset(a.dataset,a.targets,a.split_file,"validation")
    make=lambda ds,shuffle: DataLoader(ds,a.batch_size,shuffle=shuffle,num_workers=a.workers,pin_memory=device.type=="cuda",persistent_workers=a.workers>0)
    loaders={"train":make(train,True),"validation":make(val,False)}; model=GateTapNet(a.tap).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=1e-4)
    fields=["epoch","seconds","train_loss","val_loss","val_bce","val_soft_dice_loss","val_iou","val_f1","val_precision","val_recall"]; best=-1.0
    with (a.output/"log.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for epoch in range(1,a.epochs+1):
            started=time.time(); tr=run_epoch(model,loaders["train"],device,optimizer); va=run_epoch(model,loaders["validation"],device)
            row={"epoch":epoch,"seconds":time.time()-started,"train_loss":tr["loss"]}; row.update({f"val_{k}":v for k,v in va.items()}); writer.writerow(row); stream.flush()
            print(json.dumps({"tap":a.tap,"epoch":epoch,"train":tr,"validation":va}),flush=True)
            state={"epoch":epoch,"tap":a.tap,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"validation":va,"input":[1,160,160],"output":[1,40,40]}; torch.save(state,a.output/"last.pt")
            if va["iou"]>best: best=va["iou"]; torch.save(state,a.output/"best.pt")
    (a.output/"summary.json").write_text(json.dumps({"tap":a.tap,"best_validation_iou":best},indent=2)+"\n")
if __name__ == "__main__": main()
