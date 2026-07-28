#!/usr/bin/env python3
"""Generate a small, fully inspectable RGB/segmentation/bbox dataset."""

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    return parser.parse_args()


args = parse_args()
args.output_dir = args.output_dir.expanduser().resolve()
sys.argv = [sys.argv[0]]
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    launch_config={"headless": args.headless, "width": args.width, "height": args.height}
)

import carb.settings
import numpy as np
import omni.replicator.core as rep
import omni.usd
from pxr import Gf, UsdGeom, UsdLux


def main():
    omni.usd.get_context().new_stage()
    rep.orchestrator.set_capture_on_play(False)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)
    rep.functional.create.xform(name="World")
    rep.functional.create.dome_light(intensity=100, parent="/World", name="DomeLight")
    stage = omni.usd.get_context().get_stage()
    distant = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    distant.CreateIntensityAttr(800.0)
    rep.functional.create.cube(
        position=(0, 0, -1.1), scale=(20, 20, 0.2), parent="/World", name="Floor"
    )
    cube = rep.functional.create.cube(parent="/World", name="Cube")
    rep.functional.modify.semantics(cube, {"class": "cube"}, mode="add")
    cube_xform = UsdGeom.XformCommonAPI(cube)
    labels = ["cube"]

    camera = rep.functional.create.camera(
        position=(5, 5, 5), look_at=(0, 0, 0), parent="/World", name="Camera"
    )
    render_product = rep.create.render_product(
        camera, (args.width, args.height), name="TrainingRenderProduct"
    )

    backend = rep.backends.get("DiskBackend")
    backend.initialize(output_dir=str(args.output_dir))
    writer = rep.writers.get("BasicWriter")
    writer.initialize(
        backend=backend,
        rgb=True,
        bounding_box_2d_tight=True,
    )
    writer.attach(render_product)

    manifest = []
    rng = np.random.default_rng(args.seed)
    for frame in range(args.frames):
        cube_xform.SetTranslate(
            Gf.Vec3d(
                float(rng.uniform(-1.5, 1.5)),
                float(rng.uniform(-1.5, 1.5)),
                float(rng.uniform(-0.2, 0.8)),
            )
        )
        cube_xform.SetRotate(Gf.Vec3f(*(float(x) for x in rng.uniform(0, 360, 3))))
        cube_xform.SetScale(Gf.Vec3f(*(float(x) for x in rng.uniform(0.5, 1.4, 3))))
        rep.orchestrator.step()
        manifest.append({"frame": frame, "seed": args.seed, "labels": labels})

    rep.orchestrator.wait_until_complete()
    writer.detach()
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


try:
    main()
except Exception:
    import traceback

    traceback.print_exc()
    simulation_app.close()
    raise
else:
    simulation_app.close()
