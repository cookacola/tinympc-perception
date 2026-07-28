# Isaac Sim Remote Workflow Worklog

## Feature mapping

- Installation and compatibility: official Isaac Sim 6.0.1 Python bundle and compatibility checker.
- Interactive remote UI: `isaacsim.exp.full.streaming` WebRTC experience.
- Synthetic data: `data-collection-sim` workflow using Replicator `BasicWriter`.
- Headless production: `SimulationApp` with explicit off-screen RTX rendering.
- Annotation QA: programmatic validation plus one overlay image per frame.
- Training/export: local GPU training and compact `state_dict` checkpoints; rsync helper.

## Foundation results

- GPU: NVIDIA GeForce RTX 5090, 32607 MiB.
- Driver: 595.84.
- Compatibility checker: PASSED.
- Python: user-owned Python 3.12 environment at `$HOME/isaacsim-env`.
- Full Isaac Sim: 6.0.1.0 with extension caches and CUDA-enabled PyTorch.
- Docker: NVIDIA runtime configured, but user is not authorized for the Docker socket.

