# Remote workflow

## SSH and VS Code

Add this entry to `~/.ssh/config` on the laptop:

```sshconfig
Host a2r-lab
    HostName a2r-lab-server.dartmouth.edu
    User cchen
    Port 2219
```

Then connect with `ssh a2r-lab`, or choose `a2r-lab` from VS Code's
**Remote-SSH: Connect to Host** command and open:

```text
/home/cchen/isaacsim-workspace
```

GPU applications must run inside Slurm allocations, not on the login shell.

## Pull reviewed samples to the laptop

Run from the laptop:

```bash
rsync -avP -e 'ssh -p 2219' \
  cchen@a2r-lab-server.dartmouth.edu:/home/cchen/isaacsim-workspace/workspace/course_hm01b0_nbd_texture_smoke/shard_000000000/inspection_contact_sheet.jpg \
  .
```

Pull one aligned sample:

```bash
mkdir -p isaac_sample
for name in rgb_0000.png hm01b0_mono_0000.png depth_mm_0000.png \
  semantic_segmentation_0000.png semantic_segmentation_labels_0000.json \
  camera_sensor.json; do
  rsync -avP -e 'ssh -p 2219' \
    "cchen@a2r-lab-server.dartmouth.edu:/home/cchen/isaacsim-workspace/workspace/course_hm01b0_nbd_texture_smoke/shard_000000000/$name" \
    isaac_sample/
done
```

After training completes, pull only the compact checkpoint, log, and prediction
panels:

```bash
rsync -avP -e 'ssh -p 2219' \
  --include='best_total.pt' \
  --include='log.csv' \
  --include='evaluation/' \
  --include='evaluation/***' \
  --exclude='*' \
  cchen@a2r-lab-server.dartmouth.edu:/home/cchen/isaacsim-workspace/workspace/gap8_rollout_float_baseline_v3/ \
  ./gap8_rollout_float_baseline_v3/
```

Pull the QAT checkpoint and both retained/dropped-gate ONNX exports:

```bash
rsync -avP -e 'ssh -p 2219' \
  cchen@a2r-lab-server.dartmouth.edu:/home/cchen/isaacsim-workspace/workspace/gap8_rollout_qat_v3/best_qat.pt \
  cchen@a2r-lab-server.dartmouth.edu:/home/cchen/isaacsim-workspace/workspace/gap8_rollout_export_v3/ \
  ./gap8_deployment/
```

Or pull the prepared compact bundle after Slurm job `gap8_bundle` completes:

```bash
rsync -avP -e 'ssh -p 2219' \
  cchen@a2r-lab-server.dartmouth.edu:/home/cchen/isaacsim-workspace/workspace/gap8_download_bundle_v3/ \
  ./gap8_download_bundle_v3/
```

Pull one prediction panel immediately:

```bash
scp -P 2219 \
  cchen@a2r-lab-server.dartmouth.edu:/home/cchen/isaacsim-workspace/workspace/gap8_rollout_overfit_100/predictions/prediction_0000.png \
  .
```

## Interactive WebRTC

First request a GPU shell:

```bash
srun --pty -c 8 --mem=32G --gres=gpu:rtx5090:1 --time=02:00:00 bash
```

Then, from the workspace:

```bash
ISAACSIM_HOST=<compute-node-reachable-address> user_workflows/launch_webrtc.sh
```

The default signaling and streaming ports are `49100` and `47998`. Cluster
firewall/NAT policy must allow the laptop to reach the allocated compute node;
otherwise use an SSH-supported tunnel or the cluster's approved proxy.
