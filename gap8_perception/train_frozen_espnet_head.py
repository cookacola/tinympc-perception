#!/usr/bin/env python3
"""Train one frozen-ESPNet gate-head ablation."""
from __future__ import annotations
import argparse, csv, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .data import MultiTaskDataset
from .espnet_frozen_heads import FrozenESPNetHead
from .evaluate import local_centroid
from .losses import soft_dice_loss, weighted_corner_mse


def corner_step(model, batch):
    output, valid = model(batch["image"]), batch["corner_valid"]
    if model.representation == "heatmap":
        loss = weighted_corner_mse(output, batch["corners"], valid)
        prediction = local_centroid(output.sigmoid()) * 4.0
    else:
        normalized = output.sigmoid()
        loss = output.sum() * 0.0
        if valid.any(): loss = F.smooth_l1_loss(normalized[valid], batch["corner_xy"][valid] / 159.0)
        prediction = normalized * 159.0
    errors = torch.linalg.vector_norm(prediction[valid] - batch["corner_xy"][valid], dim=2) if valid.any() else None
    return loss, errors


def run_corner(model, loader, device, optimizer=None):
    model.train(optimizer is not None); loss_sum = samples = 0; errors=[]
    with (torch.enable_grad() if optimizer else torch.no_grad()):
        for batch in loader:
            batch={k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
            if optimizer: optimizer.zero_grad(set_to_none=True)
            loss,error=corner_step(model,batch)
            if optimizer: loss.backward(); optimizer.step()
            n=batch["image"].shape[0]; loss_sum += float(loss.detach())*n; samples += n
            if error is not None: errors.append(error.detach().cpu())
    e=torch.cat(errors) if errors else torch.empty(0,4)
    return {"loss":loss_sum/max(1,samples), "mean_corner_error_px":float(e.mean()) if e.numel() else float("nan"),
            "median_corner_error_px":float(e.median()) if e.numel() else float("nan"),
            "p95_corner_error_px":float(torch.quantile(e,.95)) if e.numel() else float("nan"),
            "all4_within_5px":float((e<=5).all(1).float().mean()) if e.numel() else float("nan"),
            "all4_within_10px":float((e<=10).all(1).float().mean()) if e.numel() else float("nan")}


def run_gate(model, loader, device, optimizer=None):
    model.train(optimizer is not None); totals={k:0. for k in ("loss","bce","dice","sdf_mae","intersection","union","predicted","truth")}; samples=0
    with (torch.enable_grad() if optimizer else torch.no_grad()):
        for batch in loader:
            image=batch["image"].to(device,non_blocking=True); target=batch["gate"].to(device,non_blocking=True)
            if optimizer: optimizer.zero_grad(set_to_none=True)
            logits=model(image)
            if model.representation=="binary":
                bce=F.binary_cross_entropy_with_logits(logits,target); dice=soft_dice_loss(logits,target); loss=bce+.5*dice; sdf=logits.new_zeros(()); pred=logits.sigmoid()>=.5
            else:
                sdf_target=batch["gate_sdf"].to(device,non_blocking=True); signed=logits.tanh(); loss=F.smooth_l1_loss(signed,sdf_target); sdf=F.l1_loss(signed,sdf_target); bce=dice=logits.new_zeros(()); pred=signed>=0
            if optimizer: loss.backward(); optimizer.step()
            truth=target>=.5; n=image.shape[0]; samples+=n
            for k,v in (("loss",loss),("bce",bce),("dice",dice),("sdf_mae",sdf)): totals[k]+=float(v.detach())*n
            totals["intersection"]+=float((pred&truth).sum()); totals["union"]+=float((pred|truth).sum()); totals["predicted"]+=float(pred.sum()); totals["truth"]+=float(truth.sum())
    i=totals["intersection"]
    return {"loss":totals["loss"]/samples,"bce":totals["bce"]/samples,"soft_dice_loss":totals["dice"]/samples,"sdf_mae":totals["sdf_mae"]/samples,
            "iou":i/max(1.,totals["union"]),"f1":2*i/max(1.,totals["predicted"]+totals["truth"]),"precision":i/max(1.,totals["predicted"]),"recall":i/max(1.,totals["truth"])}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--task",choices=("corner","gate"),required=True); p.add_argument("--tap",choices=("early","middle","late"),required=True)
    p.add_argument("--representation",choices=("heatmap","direct","binary","sdf"),required=True)
    for n in ("checkpoint","dataset","targets","split-file","output"): p.add_argument(f"--{n}",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=60); p.add_argument("--batch-size",type=int,default=128); p.add_argument("--workers",type=int,default=8); p.add_argument("--seed",type=int,default=20260819); p.add_argument("--learning-rate",type=float,default=2e-3); a=p.parse_args()
    allowed={"corner":{"heatmap","direct"},"gate":{"binary","sdf"}}
    if a.representation not in allowed[a.task]: p.error("representation does not match task")
    a.output.mkdir(parents=True,exist_ok=False); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets={s:MultiTaskDataset(a.dataset,a.targets,a.split_file,s) for s in ("train","validation")}
    loaders={s:DataLoader(d,a.batch_size,shuffle=s=="train",num_workers=a.workers,pin_memory=device.type=="cuda",persistent_workers=a.workers>0) for s,d in datasets.items()}
    model=FrozenESPNetHead(a.checkpoint,a.tap,a.task,a.representation).to(device); trainable=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=a.learning_rate,weight_decay=1e-4); run=run_corner if a.task=="corner" else run_gate; best=float("inf") if a.task=="corner" else -1.
    fields=None
    with (a.output/"log.csv").open("w",newline="") as stream:
        writer=None
        for epoch in range(1,a.epochs+1):
            started=time.time(); tr=run(model,loaders["train"],device,optimizer); va=run(model,loaders["validation"],device); fingerprint=model.assert_backbone_unchanged()
            row={"epoch":epoch,"seconds":time.time()-started,**{f"train_{k}":v for k,v in tr.items()},**{f"val_{k}":v for k,v in va.items()}}
            if writer is None: writer=csv.DictWriter(stream,fieldnames=list(row)); writer.writeheader()
            writer.writerow(row); stream.flush(); print(json.dumps({"task":a.task,"tap":a.tap,"representation":a.representation,"epoch":epoch,"train":tr,"validation":va,"frozen_backbone_verified":True}),flush=True)
            score=va["mean_corner_error_px"] if a.task=="corner" else va["iou"]
            state={"epoch":epoch,"task":a.task,"tap":a.tap,"representation":a.representation,"head":{k:v for k,v in model.state_dict().items() if not k.startswith("encoder.")},"validation":va,"backbone_checkpoint":str(a.checkpoint),"backbone_sha256":fingerprint,"frozen_backbone_verified":True,"temporal_input_policy":"repeat_current_frame"}; torch.save(state,a.output/"last.pt")
            improved=score<best if a.task=="corner" else score>best
            if improved: best=score; torch.save(state,a.output/"best.pt")
    (a.output/"summary.json").write_text(json.dumps({"task":a.task,"tap":a.tap,"representation":a.representation,"best_score":best,"score_name":"mean_corner_error_px" if a.task=="corner" else "iou","backbone_checkpoint":str(a.checkpoint),"backbone_sha256":model.assert_backbone_unchanged(),"frozen_backbone_verified":True,"temporal_input_policy":"repeat_current_frame"},indent=2)+"\n")
if __name__=="__main__": main()
