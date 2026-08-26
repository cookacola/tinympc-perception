#!/usr/bin/env python3
"""Generate one resumable shard of aligned racetrack RGB/depth/semantics."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

# Isaac Kit owns its worker pool.  Avoid initializing large, competing BLAS
# pools before Kit starts; repeated cold launches otherwise occasionally wait
# forever on a pre-CUDA futex on a2r-main.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import cv2
import numpy as np

OVERSCAN_RESOLUTION = 512
LIGHTING_PROFILES = (
    ("neutral", 120.0, 900.0, 5200.0),
    # Keep the low-light profile visibly dim without producing near-black
    # HM01B0 frames that would not be useful for gate or obstacle supervision.
    ("dim", 90.0, 650.0, 4800.0),
    ("low_contrast", 240.0, 500.0, 5200.0),
    ("strong_shadow", 70.0, 1500.0, 5400.0),
    ("warm", 130.0, 1150.0, 3400.0),
    ("cool", 130.0, 1050.0, 7200.0),
    ("backlit", 80.0, 1350.0, 6200.0),
)
BACKGROUND_PROFILES = (
    ("warm_wood_lab", (0.35, 0.37, 0.39), (0.50, 0.24, 0.08), (0.68, 0.36, 0.12), (0.72, 0.70, 0.66), (0.18, 0.32, 0.58)),
    ("cool_gray_workshop", (0.22, 0.24, 0.27), (0.34, 0.37, 0.40), (0.52, 0.55, 0.58), (0.43, 0.48, 0.54), (0.75, 0.33, 0.12)),
    ("blue_mat_lab", (0.20, 0.21, 0.23), (0.08, 0.19, 0.33), (0.12, 0.32, 0.48), (0.70, 0.73, 0.76), (0.76, 0.63, 0.10)),
    ("beige_classroom", (0.42, 0.40, 0.36), (0.55, 0.48, 0.36), (0.73, 0.67, 0.54), (0.78, 0.74, 0.64), (0.18, 0.45, 0.28)),
    ("high_contrast_tiles", (0.18, 0.18, 0.18), (0.12, 0.12, 0.12), (0.78, 0.78, 0.75), (0.50, 0.52, 0.55), (0.55, 0.12, 0.16)),
)
COURSE_FAMILIES_BY_SPLIT = {
    "train": ("open_field", "slalom", "corridor", "chicane", "mixed_primitives", "dense_clutter"),
    "validation": ("offset_rooms", "pillar_forest"),
    "test": ("warehouse_aisles", "wall_maze"),
}
TEXTURE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lighting-scale",
        type=float,
        default=1.0,
        help="Multiply all dome and sun intensities without changing camera exposure.",
    )
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument(
        "--rt-subframes",
        type=int,
        default=1,
        help="RTX accumulation subframes per captured frame (use >1 for material smoke tests)",
    )
    parser.add_argument(
        "--camera-calibration",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "gap8_perception/configs/hm01b0_calibration.json",
    )
    parser.add_argument(
        "--gate-texture-version",
        choices=("v1", "hm01b0_v2", "photo_v1"),
        default="photo_v1",
    )
    parser.add_argument(
        "--gate-target-probability",
        type=float,
        default=0.5,
        help="Probability that a sampled camera targets one of the two gates.",
    )
    parser.add_argument(
        "--no-gates",
        action="store_true",
        help="Omit all gate geometry and sample only obstacle/background views.",
    )
    parser.add_argument(
        "--counterfactual-gate-camera-policy",
        action="store_true",
        help=(
            "With --no-gates, retain the gate-scene camera sampling policy. "
            "Using the same seed then produces paired images that differ only "
            "by the omitted gate geometry."
        ),
    )
    parser.add_argument(
        "--layout-family",
        choices=("fixed_v1", "multi_course_v2"),
        default="multi_course_v2",
    )
    parser.add_argument(
        "--asset-split",
        choices=("train", "validation", "test"),
        default="train",
        help="Select disjoint course families and internet texture assets.",
    )
    parser.add_argument(
        "--clutter-texture-manifest",
        type=Path,
        help="Manifest produced by download_commons_clutter_textures.py.",
    )
    return parser.parse_args()


args = parse_args()
args.output_dir = args.output_dir.expanduser().resolve()
args.camera_calibration = args.camera_calibration.expanduser().resolve()
if args.clutter_texture_manifest is not None:
    args.clutter_texture_manifest = args.clutter_texture_manifest.expanduser().resolve()
if not 0.0 <= args.gate_target_probability <= 1.0:
    raise ValueError("gate target probability must be in [0, 1]")
if args.counterfactual_gate_camera_policy and not args.no_gates:
    raise ValueError("--counterfactual-gate-camera-policy requires --no-gates")
if args.lighting_scale <= 0:
    raise ValueError("lighting scale must be positive")
camera_calibration = json.loads(args.camera_calibration.read_text())
background_profile = BACKGROUND_PROFILES[(args.start_index // max(args.frames, 1)) % len(BACKGROUND_PROFILES)]


def sample_layout():
    if args.layout_family == "fixed_v1":
        return {
            "family": "fixed_v1",
            "seed": None,
            "gates": [(-1.30, -0.45, 0.55), (1.30, 0.45, 0.55)],
            "obstacles": [
                {"primitive": "box", "center": (-0.55, 0.65, 0.30), "size": (0.55, 0.55, 0.60)},
                {"primitive": "box", "center": (0.60, -0.65, 0.38), "size": (0.50, 0.75, 0.76)},
            ],
        }
    layout_seed = args.seed + args.start_index
    rng = np.random.default_rng(layout_seed)
    families = COURSE_FAMILIES_BY_SPLIT[args.asset_split]
    family = families[(args.start_index // max(args.frames, 1)) % len(families)]
    gates = [
        (-1.30, float(rng.uniform(-0.75, 0.75)), float(rng.uniform(0.48, 0.66))),
        (1.30, float(rng.uniform(-0.75, 0.75)), float(rng.uniform(0.48, 0.66))),
    ]
    obstacles = []

    def add(center, size, primitive="box"):
        center = tuple(float(value) for value in center)
        size = tuple(float(value) for value in size)
        obstacles.append({"primitive": primitive, "center": center, "size": size})

    def add_random(count, primitives=("box", "cylinder", "cone", "sphere")):
        attempts = 0
        while count > 0 and attempts < 1000:
            attempts += 1
            primitive = str(rng.choice(primitives))
            sx, sy = rng.uniform(0.24, 0.78, size=2)
            if primitive in {"sphere", "cylinder", "cone"}:
                sy = sx
            sz = float(rng.uniform(0.28, 1.20))
            x, y = rng.uniform(-1.35, 1.35, size=2)
            if min(np.hypot(x - gate[0], y - gate[1]) for gate in gates) < 0.58 + max(sx, sy) / 2:
                continue
            if any(
                abs(x - item["center"][0]) < (sx + item["size"][0]) / 2 + 0.10
                and abs(y - item["center"][1]) < (sy + item["size"][1]) / 2 + 0.10
                for item in obstacles
            ):
                continue
            add((x, y, sz / 2), (sx, sy, sz), primitive)
            count -= 1
        if count:
            raise RuntimeError(f"could not place all obstacles for {family}")

    if family == "open_field":
        add_random(int(rng.integers(4, 7)))
    elif family == "slalom":
        for index, x in enumerate(np.linspace(-0.85, 0.85, 5)):
            diameter = float(rng.uniform(0.28, 0.52))
            height = float(rng.uniform(0.55, 1.15))
            add((x, (-1) ** index * rng.uniform(0.42, 0.82), height / 2),
                (diameter, diameter, height), str(rng.choice(("cylinder", "cone"))))
    elif family == "corridor":
        gap = float(rng.uniform(0.75, 1.15))
        for side in (-1, 1):
            add((0.0, side * (gap / 2 + 0.34), 0.55), (2.2, 0.55, 1.10))
        add_random(2, ("box", "cylinder"))
    elif family == "chicane":
        side = int(rng.choice((-1, 1)))
        add((-0.55, -side * 0.78, 0.65), (0.28, 1.65, 1.30))
        add((0.55, side * 0.78, 0.65), (0.28, 1.65, 1.30))
        add_random(2, ("box", "cylinder"))
    elif family == "mixed_primitives":
        add_random(int(rng.integers(6, 9)))
    elif family == "dense_clutter":
        add_random(int(rng.integers(9, 13)), ("box", "cylinder", "cone"))
    elif family == "offset_rooms":
        add((-0.45, -1.05, 0.75), (0.24, 1.65, 1.50))
        add((0.45, 1.05, 0.75), (0.24, 1.65, 1.50))
        add_random(4, ("box", "cylinder"))
    elif family == "pillar_forest":
        for x, y in ((-0.75, -0.75), (-0.75, 0.75), (0.0, 0.0), (0.75, -0.75), (0.75, 0.75)):
            diameter = float(rng.uniform(0.24, 0.42))
            height = float(rng.uniform(0.70, 1.35))
            add((x, y, height / 2), (diameter, diameter, height), "cylinder")
    elif family == "warehouse_aisles":
        for x in (-0.62, 0.62):
            for y in (-1.05, 1.05):
                add((x, y, 0.72), (0.45, 1.05, 1.44))
        add_random(3, ("box",))
    elif family == "wall_maze":
        add((-0.70, -0.90, 0.72), (0.22, 1.55, 1.44))
        add((0.0, 0.90, 0.72), (0.22, 1.55, 1.44))
        add((0.70, -0.90, 0.72), (0.22, 1.55, 1.44))
        add_random(3, ("box", "cylinder"))
    else:
        raise ValueError(f"unknown course family: {family}")
    return {"family": family, "seed": layout_seed, "gates": gates, "obstacles": obstacles}


def texture_assets():
    if args.clutter_texture_manifest is None:
        return []
    manifest = json.loads(args.clutter_texture_manifest.read_text())
    root = args.clutter_texture_manifest.parent
    selected = []
    for asset in manifest.get("assets", []):
        path = root / asset["file"]
        if (
            asset.get("catalog") == "Openverse"
            and asset.get("split") == args.asset_split
            and path.suffix.lower() in TEXTURE_EXTENSIONS
            and path.is_file()
        ):
            selected.append({**asset, "path": path})
    return selected


LAYOUT = sample_layout()
CLUTTER_TEXTURES = texture_assets()
sys.argv = [sys.argv[0]]
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp

app = SimulationApp(
    launch_config={
        "headless": True,
        "width": args.width,
        "height": args.height,
        "renderer": "RayTracedLighting",
        "multi_gpu": False,
        "active_gpu": 0,
        "physics_gpu": 0,
    }
)

import carb.settings
import omni.kit.app
import omni.replicator.core as rep
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt

CLASS_NAMES = ["background", "course", "boundary", "obstacle", "gate", "lab_clutter"]
ASSET_ROOT = Path(__file__).resolve().parents[1] / "gates"
GATE_TEXTURE_DIR = (
    ASSET_ROOT
    if args.gate_texture_version == "v1"
    else ASSET_ROOT / "newbeedrone_hm01b0_v2"
)
GATE_TEXTURE_SUFFIX = "v1" if args.gate_texture_version == "v1" else "v2"
PHOTO_TEXTURE_PATH = (
    ASSET_ROOT
    / "newbeedrone_photo_reference_v1"
    / "newbeedrone_gate_front_uv_v1.png"
)


def create_box(
    name, position, scale, label, parent="/World/Course", color=None, material=None
):
    prim = rep.functional.create.cube(
        position=position, scale=scale, parent=parent, name=name, material=material
    )
    rep.functional.modify.semantics(prim, {"class": label}, mode="add")
    if color is not None:
        UsdGeom.Gprim(prim).CreateDisplayColorAttr(
            [Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))]
        )
    return prim


def create_obstacle(name, spec, material=None, color=None):
    creators = {
        "box": rep.functional.create.cube,
        "sphere": rep.functional.create.sphere,
        "cylinder": rep.functional.create.cylinder,
        "cone": rep.functional.create.cone,
    }
    primitive = spec["primitive"]
    prim = creators[primitive](
        position=spec["center"], scale=spec["size"], parent="/World/Course",
        name=name, material=material,
    )
    rep.functional.modify.semantics(prim, {"class": "obstacle"}, mode="add")
    if color is not None and material is None:
        UsdGeom.Gprim(prim).CreateDisplayColorAttr(
            [Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))]
        )
    return prim


def create_scene_material(name, texture, color, roughness, texture_scale):
    values = {
        "mdl": "OmniPBR.mdl",
        "diffuse_color_constant": tuple(float(value) for value in color),
        "reflection_roughness_constant": float(roughness),
        "name": name,
        "parent": "/World",
    }
    if texture is not None:
        values.update(
            diffuse_texture=str(texture),
            project_uvw=True,
            texture_scale=(float(texture_scale), float(texture_scale)),
        )
    return rep.functional.create.material(**values)


def create_gate_face(
    path, x, y, z, width_y, height_z, material
):
    """Create one explicitly UV-mapped Y-Z fabric quad."""
    stage = omni.usd.get_context().get_stage()
    mesh = UsdGeom.Mesh.Define(stage, path)
    half_width, half_height = width_y / 2.0, height_z / 2.0
    mesh.CreatePointsAttr(
        Vt.Vec3fArray(
            [
                (x, y - half_width, z - half_height),
                (x, y + half_width, z - half_height),
                (x, y + half_width, z + half_height),
                (x, y - half_width, z + half_height),
            ]
        )
    )
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    st.Set(Vt.Vec2fArray([(0, 1), (1, 1), (1, 0), (0, 0)]))
    UsdShade.MaterialBindingAPI(mesh).Bind(UsdShade.Material(material))
    rep.functional.modify.semantics(mesh.GetPrim(), {"class": "gate"}, mode="add")
    return mesh.GetPrim()


def rounded_square_points(half_extent, radius, segments_per_corner=8):
    """Return a counter-clockwise rounded-square contour in local Y-Z."""
    if not 0.0 < radius < half_extent:
        raise ValueError("rounded-square radius must be inside its half extent")
    centers_and_angles = (
        ((half_extent - radius, half_extent - radius), (0.0, math.pi / 2.0)),
        ((-half_extent + radius, half_extent - radius), (math.pi / 2.0, math.pi)),
        ((-half_extent + radius, -half_extent + radius), (math.pi, 3.0 * math.pi / 2.0)),
        ((half_extent - radius, -half_extent + radius), (3.0 * math.pi / 2.0, 2.0 * math.pi)),
    )
    points = []
    for (center_y, center_z), (start, stop) in centers_and_angles:
        for angle in np.linspace(start, stop, segments_per_corner, endpoint=False):
            points.append(
                (
                    center_y + radius * math.cos(float(angle)),
                    center_z + radius * math.sin(float(angle)),
                )
            )
    return points


def create_textured_gate_ring(
    path,
    x,
    y_center,
    z_center,
    outer_size,
    inner_size,
    material,
    mirror_u=False,
):
    """Create one UV-mapped rounded-ring face using the approved full-gate atlas."""
    stage = omni.usd.get_context().get_stage()
    mesh = UsdGeom.Mesh.Define(stage, path)
    outer = rounded_square_points(outer_size / 2.0, radius=0.090)
    inner = rounded_square_points(inner_size / 2.0, radius=0.075)
    points = []
    st_values = []
    for contour in (outer, inner):
        for local_y, local_z in contour:
            points.append((x, y_center + local_y, z_center + local_z))
            u = 0.5 + local_y / outer_size
            if mirror_u:
                u = 1.0 - u
            st_values.append((u, 0.5 - local_z / outer_size))
    count = len(outer)
    face_indices = []
    for index in range(count):
        following = (index + 1) % count
        face_indices.extend((index, following, count + following, count + index))
    mesh.CreatePointsAttr(Vt.Vec3fArray(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * count))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(face_indices))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    st.Set(Vt.Vec2fArray(st_values))
    UsdShade.MaterialBindingAPI(mesh).Bind(UsdShade.Material(material))
    rep.functional.modify.semantics(mesh.GetPrim(), {"class": "gate"}, mode="add")
    return mesh.GetPrim()


def create_gate_edge_walls(
    path,
    x_center,
    y_center,
    z_center,
    depth,
    outer_size,
    inner_size,
    material,
):
    """Give the rounded fabric ring physical depth for oblique camera views."""
    stage = omni.usd.get_context().get_stage()
    mesh = UsdGeom.Mesh.Define(stage, path)
    contours = (
        rounded_square_points(outer_size / 2.0, radius=0.090),
        rounded_square_points(inner_size / 2.0, radius=0.075),
    )
    points = []
    face_indices = []
    face_count = 0
    for contour in contours:
        base = len(points)
        for x_offset in (-depth / 2.0, depth / 2.0):
            points.extend(
                (x_center + x_offset, y_center + local_y, z_center + local_z)
                for local_y, local_z in contour
            )
        count = len(contour)
        for index in range(count):
            following = (index + 1) % count
            face_indices.extend(
                (base + index, base + following, base + count + following, base + count + index)
            )
            face_count += 1
    mesh.CreatePointsAttr(Vt.Vec3fArray(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4] * face_count))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(face_indices))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI(mesh).Bind(UsdShade.Material(material))
    rep.functional.modify.semantics(mesh.GetPrim(), {"class": "gate"}, mode="add")
    return mesh.GetPrim()


def look_at_matrix(eye, target):
    eye = Gf.Vec3d(*eye)
    target = Gf.Vec3d(*target)
    up = Gf.Vec3d(0, 0, 1)
    forward = (target - eye).GetNormalized()
    if abs(forward * up) > 0.99:
        up = Gf.Vec3d(0, 1, 0)
    right = (forward ^ up).GetNormalized()
    camera_up = (right ^ forward).GetNormalized()
    matrix = Gf.Matrix4d()
    matrix[0] = [right[0], right[1], right[2], 0]
    matrix[1] = [camera_up[0], camera_up[1], camera_up[2], 0]
    matrix[2] = [-forward[0], -forward[1], -forward[2], 0]
    matrix[3] = [eye[0], eye[1], eye[2], 1]
    return matrix


def build_scene():
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    rep.orchestrator.set_capture_on_play(False)
    settings = carb.settings.get_settings()
    settings.set("rtx/post/dlss/execMode", 2)
    settings.set("/rtx/post/tonemap/op", 4)
    settings.set("/rtx/post/tonemap/filmIso", 200)

    rep.functional.create.xform(name="World")
    rep.functional.create.xform(parent="/World", name="Course")
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(120.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(900.0)
    sun.CreateAngleAttr(1.0)
    sun.CreateEnableColorTemperatureAttr(True)
    sun.CreateColorTemperatureAttr(5200.0)
    sun_xform = UsdGeom.Xformable(sun.GetPrim())
    sun_rotate_x = sun_xform.AddRotateXOp()
    sun_rotate_z = sun_xform.AddRotateZOp()

    profile_name, floor_color, tile_a, tile_b, wall_color, accent_color = background_profile
    appearance_rng = np.random.default_rng((LAYOUT["seed"] or args.seed) + 700001)
    texture_order = list(CLUTTER_TEXTURES)
    appearance_rng.shuffle(texture_order)
    scene_materials = [
        create_scene_material(
            f"InternetTexture_{index}", asset["path"], (1.0, 1.0, 1.0),
            appearance_rng.uniform(0.62, 0.96), appearance_rng.uniform(0.55, 2.4),
        )
        for index, asset in enumerate(texture_order[:16])
    ]
    # Deterministic per-shard materials prevent the no-gate classifier from
    # learning one room or floor texture as its negative shortcut.
    create_box(
        "LabFloor", (0, 0, -0.08), (8, 8, 0.12), "lab_clutter",
        parent="/World", color=floor_color,
    )
    for row in range(8):
        for column in range(8):
            x = -1.75 + column * 0.5
            y = -1.75 + row * 0.5
            if LAYOUT["family"] in {"corridor", "warehouse_aisles"}:
                palette_index = column % 2
            elif LAYOUT["family"] in {"chicane", "wall_maze"}:
                palette_index = row % 2
            elif LAYOUT["family"] == "dense_clutter":
                palette_index = int(appearance_rng.integers(0, 2))
            else:
                palette_index = (row + column) % 2
            wood = (tile_a, tile_b)[palette_index]
            create_box(
                f"WoodTile_{row}_{column}", (x, y, 0.0), (0.49, 0.49, 0.04),
                "course", color=wood,
            )
    for name, position, scale in [
        ("NorthTrim", (0, 2.0, 0.04), (4.1, 0.06, 0.08)),
        ("SouthTrim", (0, -2.0, 0.04), (4.1, 0.06, 0.08)),
        ("WestTrim", (-2.0, 0, 0.04), (0.06, 4.1, 0.08)),
        ("EastTrim", (2.0, 0, 0.04), (0.06, 4.1, 0.08)),
    ]:
        create_box(name, position, scale, "boundary", color=(0.12, 0.12, 0.12))

    for wall_index, (name, position, scale) in enumerate((
        ("Backdrop_N", (0.0, 4.0, 1.5), (8.0, 0.10, 3.0)),
        ("Backdrop_W", (-4.0, 0.0, 1.5), (0.10, 8.0, 3.0)),
        ("Backdrop_E", (4.0, 0.0, 1.5), (0.10, 8.0, 3.0)),
    )):
        material = scene_materials[wall_index % len(scene_materials)] if scene_materials else None
        create_box(
            name, position, scale, "lab_clutter", parent="/World",
            color=None if material is not None else wall_color, material=material,
        )
    # Repeated panels produce different high-frequency background textures at
    # the HM01B0 resolution without introducing gate-like square openings.
    for panel in range(10):
        y = -3.2 + 0.70 * panel
        create_box(f"WallAccent_{panel}", (-3.91, y, 1.0 + 0.35 * (panel % 3)),
                   (0.04, 0.32, 0.18), "lab_clutter", parent="/World", color=accent_color)

    obstacle_colors = ((0.12, 0.28, 0.65), (0.68, 0.14, 0.10), (0.18, 0.58, 0.24),
                       (0.62, 0.54, 0.16), (0.42, 0.23, 0.55))
    for obstacle_index, obstacle in enumerate(LAYOUT["obstacles"]):
        material = (
            scene_materials[(obstacle_index + 3) % len(scene_materials)]
            if scene_materials and appearance_rng.random() < 0.75 else None
        )
        create_obstacle(
            f"Obstacle_{obstacle_index}", obstacle, material=material,
            color=None if material is not None else obstacle_colors[obstacle_index % len(obstacle_colors)],
        )

    # Exactly two NewBeeDrone Micro Race Gate - Square assemblies.
    # Commercial listings describe an approximately 0.66 m outer square and
    # 0.45 m clear opening, giving a 0.105 m fabric-frame width.
    gate_outer = 0.66
    gate_inner = 0.45
    gate_frame = (gate_outer - gate_inner) / 2.0
    gate_offset = gate_inner / 2.0 + gate_frame / 2.0
    gate_center_z = 0.55
    gate_base_material = rep.functional.create.material(
        mdl="OmniPBR.mdl",
        diffuse_color_constant=(0.10, 0.10, 0.10),
        reflection_roughness_constant=0.88,
        name="NewBeeDrone_Base_Fabric",
        parent="/World",
    )
    gate_materials = {}
    if args.gate_texture_version == "photo_v1":
        if not PHOTO_TEXTURE_PATH.is_file():
            raise FileNotFoundError(f"missing approved gate texture: {PHOTO_TEXTURE_PATH}")
        photo_material = rep.functional.create.material(
            mdl="OmniPBR.mdl",
            diffuse_texture=str(PHOTO_TEXTURE_PATH),
            diffuse_color_constant=(1.0, 1.0, 1.0),
            reflection_roughness_constant=0.82,
            project_uvw=False,
            name="NewBeeDrone_Approved_Photo_Material",
            parent="/World",
        )
    else:
        for part in ("top", "bottom", "left", "right"):
            texture_path = (
                GATE_TEXTURE_DIR / f"newbeedrone_{part}_{GATE_TEXTURE_SUFFIX}.png"
            )
            if not texture_path.is_file():
                raise FileNotFoundError(f"missing gate texture: {texture_path}")
            gate_materials[part] = rep.functional.create.material(
                mdl="OmniPBR.mdl",
                diffuse_texture=str(texture_path),
                diffuse_color_constant=(1.0, 1.0, 1.0),
                reflection_roughness_constant=0.82,
                project_uvw=False,
                name=f"NewBeeDrone_{part.title()}_Material",
                parent="/World",
            )
    gate_positions = () if args.no_gates else LAYOUT["gates"]
    for gate_index, (x_pos, y_center, gate_center_z) in enumerate(gate_positions):
        gate_parent = f"/World/Course/Gate_{gate_index}"
        rep.functional.create.xform(parent="/World/Course", name=f"Gate_{gate_index}")
        if args.gate_texture_version == "photo_v1":
            depth = 0.025
            create_textured_gate_ring(
                f"{gate_parent}/ApprovedFront",
                x_pos - depth / 2.0,
                y_center,
                gate_center_z,
                gate_outer,
                gate_inner,
                photo_material,
            )
            create_textured_gate_ring(
                f"{gate_parent}/ApprovedBack",
                x_pos + depth / 2.0,
                y_center,
                gate_center_z,
                gate_outer,
                gate_inner,
                photo_material,
                mirror_u=True,
            )
            create_gate_edge_walls(
                f"{gate_parent}/FabricEdges",
                x_pos,
                y_center,
                gate_center_z,
                depth,
                gate_outer,
                gate_inner,
                gate_base_material,
            )
        else:
            for side, y_pos in (("Left", -gate_offset), ("Right", gate_offset)):
                create_box(
                    side,
                    (x_pos, y_center + y_pos, gate_center_z),
                    (0.025, gate_frame, gate_outer),
                    "gate",
                    parent=gate_parent,
                    material=gate_base_material,
                )
            for edge, z_pos in (
                ("Bottom", gate_center_z - gate_offset),
                ("Top", gate_center_z + gate_offset),
            ):
                create_box(
                    edge,
                    (x_pos, y_center, z_pos),
                    (0.025, gate_outer, gate_frame),
                    "gate",
                    parent=gate_parent,
                    material=gate_base_material,
                )

            face_epsilon = 0.018
            face_specs = [
                ("left", y_center - gate_offset, gate_center_z,
                 (gate_frame, 1.0, gate_inner)),
                ("right", y_center + gate_offset, gate_center_z,
                 (gate_frame, 1.0, gate_inner)),
                ("bottom", y_center, gate_center_z - gate_offset,
                 (gate_outer, 1.0, gate_frame)),
                ("top", y_center, gate_center_z + gate_offset,
                 (gate_outer, 1.0, gate_frame)),
            ]
            for part, y_pos, z_pos, scale in face_specs:
                for face_index, x_offset in enumerate((-face_epsilon, face_epsilon)):
                    create_gate_face(
                        f"{gate_parent}/{part.title()}_FabricFace_{face_index}",
                        x_pos + x_offset,
                        y_pos,
                        z_pos,
                        scale[0],
                        scale[2],
                        gate_materials[part],
                    )

    # Lab clutter remains outside the 4 m x 4 m course.
    rep.functional.create.xform(parent="/World", name="LabClutter")
    clutter_parent = "/World/LabClutter"
    for name, position, scale, color in [
        ("TableTop_W", (-2.75, 0.0, 0.78), (1.1, 2.0, 0.10), (0.30, 0.18, 0.08)),
        ("TableTop_E", (2.75, 0.1, 0.78), (1.1, 1.8, 0.10), (0.30, 0.18, 0.08)),
        ("Computer_W", (-2.70, 0.0, 1.15), (0.12, 0.65, 0.48), accent_color),
        ("Computer_E", (2.70, 0.1, 1.15), (0.12, 0.65, 0.48), accent_color),
        ("Shelf_Back", (0.0, 3.10, 1.10), (3.0, 0.45, 2.2), (0.25, 0.27, 0.30)),
        ("Shelf_Side", (-3.25, -2.7, 1.00), (0.60, 1.5, 2.0), (0.25, 0.27, 0.30)),
    ]:
        create_box(
            name, position, scale, "lab_clutter", parent=clutter_parent, color=color
        )

    camera = rep.functional.create.camera(
        position=(0, -12, 4), look_at=(0, 0, 0), parent="/World", name="Camera"
    )
    intrinsic = camera_calibration["camera_matrix"]
    usd_camera = UsdGeom.Camera(camera)
    aperture = 20.955
    focal_length = float(intrinsic[0][0] * aperture / OVERSCAN_RESOLUTION)
    usd_camera.CreateHorizontalApertureAttr(aperture)
    usd_camera.CreateVerticalApertureAttr(
        float(focal_length * OVERSCAN_RESOLUTION / intrinsic[1][1])
    )
    usd_camera.CreateFocalLengthAttr(focal_length)
    sensor = camera_calibration["sensor_settings"]
    camera.CreateAttribute("omni:rtx:autoExposure:enabled", Sdf.ValueTypeNames.Bool).Set(
        bool(sensor["auto_exposure"])
    )
    camera.CreateAttribute("exposure:time", Sdf.ValueTypeNames.Float).Set(
        float(sensor["integration_time_ms"]) / 1000.0
    )
    camera.CreateAttribute("exposure:iso", Sdf.ValueTypeNames.Float).Set(
        100.0 * float(sensor["analog_gain"]) * float(sensor["digital_gain"])
    )
    camera_xform = UsdGeom.Xformable(camera)
    camera_xform.ClearXformOpOrder()
    camera_matrix = camera_xform.MakeMatrixXform()
    render_product = rep.create.render_product(
        camera, (OVERSCAN_RESOLUTION, OVERSCAN_RESOLUTION), name="CourseRenderProduct"
    )
    return camera_matrix, render_product, dome, sun, sun_rotate_x, sun_rotate_z


def distortion_remap():
    intrinsic = np.asarray(camera_calibration["camera_matrix"], np.float64)
    distortion = np.asarray(
        camera_calibration.get("simulation_distortion_coefficients",
                               camera_calibration["distortion_coefficients"]),
        np.float64,
    )
    u, v = np.meshgrid(np.arange(args.width, dtype=np.float64),
                       np.arange(args.height, dtype=np.float64))
    pixels = np.stack((u, v), axis=-1).reshape(-1, 1, 2)
    ideal = cv2.undistortPointsIter(
        pixels, intrinsic, distortion, None, None,
        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12),
    ).reshape(args.height, args.width, 2)
    return ((ideal[..., 0] * intrinsic[0, 0] + OVERSCAN_RESOLUTION / 2).astype(np.float32),
            (ideal[..., 1] * intrinsic[1, 1] + OVERSCAN_RESOLUTION / 2).astype(np.float32))


def apply_sensor_model(output_dir, seed):
    """Calibrated pinhole remap followed by deterministic HM01B0-like sampling."""
    map_x, map_y = distortion_remap()
    rng = np.random.default_rng(seed)
    for source in sorted(output_dir.glob("rgb_*.png")):
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        response = np.clip(0.04 + 0.96 * np.power(gray, 1.15), 0.0, 1.0)
        noise = rng.normal(0.0, np.sqrt(10.0 + 45.0 * response), response.shape)
        sampled = np.clip(np.rint(response * 255.0 + noise), 0, 255).astype(np.uint8)
        cv2.imwrite(str(source), cv2.cvtColor(sampled, cv2.COLOR_GRAY2BGR))
    for source in sorted(output_dir.glob("distance_to_image_plane_*.npy")):
        depth = np.load(source, allow_pickle=False)
        np.save(source, cv2.remap(depth, map_x, map_y, cv2.INTER_NEAREST), allow_pickle=False)
    for source in sorted(output_dir.glob("semantic_segmentation_*.png")):
        semantic = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        cv2.imwrite(str(source), cv2.remap(semantic, map_x, map_y, cv2.INTER_NEAREST))


def sample_camera(rng):
    # Aim primarily at course features so the physically small micro gates and
    # varied obstacles are well represented while retaining broad pose diversity.
    obstacle_points = [item["center"] for item in LAYOUT["obstacles"]]
    if args.no_gates and not args.counterfactual_gate_camera_policy:
        points = np.asarray(obstacle_points)
        point_index = int(rng.integers(0, len(points)))
        targets_gate = False
    else:
        gate_count = len(LAYOUT["gates"])
        points = np.asarray(LAYOUT["gates"] + obstacle_points)
        if rng.random() < args.gate_target_probability:
            point_index = int(rng.integers(0, gate_count))
        else:
            point_index = int(rng.integers(gate_count, len(points)))
        targets_gate = point_index < gate_count
    target = points[point_index].copy()
    target += rng.normal(
        (0, 0, 0), (0.06, 0.06, 0.05) if targets_gate else (0.20, 0.20, 0.12)
    )
    if targets_gate:
        # The gate plane is Y-Z: approach it primarily along X so the square
        # opening, rather than its 25 mm edge, is visible. Approach each gate
        # from the course interior; approaching from the outside would clamp
        # the eye against the course boundary and put the fabric frame almost
        # on the camera.
        # Keep the 0.555 m label square roughly 51--88 px wide at fx=183 px:
        # large enough for corner regression without near-field cropping.
        distance = float(rng.uniform(1.15, 2.0))
        inward_azimuth = math.pi if LAYOUT["gates"][point_index][0] < 0 else 0.0
        azimuth_offset = float(rng.uniform(-math.radians(18), math.radians(18)))
        azimuth = inward_azimuth + azimuth_offset
    else:
        # Keep the camera outside the two obstacle volumes.
        distance = float(rng.uniform(1.8, 2.8))
        azimuth = float(rng.uniform(-math.pi, math.pi))
    x = float(np.clip(target[0] - distance * math.cos(azimuth), -1.85, 1.85))
    y = float(np.clip(target[1] - distance * math.sin(azimuth), -1.85, 1.85))
    if targets_gate:
        target[2] = float(np.clip(target[2], 0.42, 0.62))
        elevation = float(rng.uniform(math.radians(-12), math.radians(18)))
        z = float(target[2] + distance * math.tan(elevation))
    else:
        target[2] = float(np.clip(target[2], 0.18, 0.62))
        z = float(np.clip(target[2] + rng.uniform(0.25, 1.00), 0.48, 1.50))
    # Move a camera that landed inside an obstacle toward its target until it
    # clears the obstacle's conservative axis-aligned envelope.
    eye = np.asarray((x, y, z), dtype=float)
    direction = np.asarray(target, dtype=float) - eye
    norm = np.linalg.norm(direction)
    if norm > 1e-6:
        direction /= norm
    for _ in range(20):
        collides = any(
            np.all(np.abs(eye - np.asarray(item["center"])) <= np.asarray(item["size"]) / 2 + 0.08)
            for item in LAYOUT["obstacles"]
        )
        if not collides:
            break
        eye += direction * 0.10
    return tuple(float(value) for value in eye), tuple(float(v) for v in target)


def main():
    camera_matrix, render_product, dome, sun, sun_rotate_x, sun_rotate_z = build_scene()
    backend = rep.backends.get("DiskBackend")
    backend.initialize(output_dir=str(args.output_dir))
    (args.output_dir / "scene_metadata.json").write_text(
        json.dumps(
            {
                "course_count": 1,
                "course_extent_m": [4.0, 4.0],
                "floor": "alternating wooden tiles",
                "background_profile": background_profile[0],
                "background_profiles_available": [profile[0] for profile in BACKGROUND_PROFILES],
                "course_family": LAYOUT["family"],
                "course_family_split": args.asset_split,
                "course_families_available": COURSE_FAMILIES_BY_SPLIT,
                "layout_seed": LAYOUT["seed"],
                "obstacles": LAYOUT["obstacles"],
                "obstacle_count": len(LAYOUT["obstacles"]),
                "gate_count": 0 if args.no_gates else 2,
                "gate_model": None if args.no_gates else "NewBeeDrone Micro Race Gate - Square",
                "gate_outer_size_m": [0.66, 0.66],
                "gate_clear_opening_m": [0.45, 0.45],
                "gate_frame_width_m": 0.105,
                "gate_corner_supervision": {
                    "definition": "midpoint between corresponding inner and outer corners",
                    "corner_square_size_m": [0.555, 0.555],
                    "matches": "real-flight labeled corner convention",
                },
                "gate_texture": {
                    "version": f"newbeedrone_{args.gate_texture_version}",
                    "appearance": "black fabric with HM01B0-visible honeycomb panels, seams, and white NewBeeDrone branding",
                    "atlas": (
                        "gates/newbeedrone_photo_reference_v1/newbeedrone_gate_front_uv_v1.png"
                        if args.gate_texture_version == "photo_v1"
                        else f"gates/newbeedrone_{args.gate_texture_version}"
                    ),
                    "source": (
                        "approved product-photo reference"
                        if args.gate_texture_version == "photo_v1"
                        else "legacy renderer texture"
                    ),
                },
                "geometry": (
                    "rounded extruded UV ring"
                    if args.gate_texture_version == "photo_v1"
                    else "fixed box rails"
                ),
                "outside_context": ["tables", "computers", "shelves"],
                "randomized": ["camera_position", "camera_look_at", "lighting_profile",
                               "light_intensity", "light_temperature", "light_direction",
                               "course_family", "obstacle_position", "obstacle_size",
                               "obstacle_primitive", "floor_pattern", "wall_texture",
                               "obstacle_texture", "floor_palette", "wall_palette",
                               "clutter_palette", "wall_panels"],
                "clutter_texture_manifest": (
                    str(args.clutter_texture_manifest) if args.clutter_texture_manifest else None
                ),
                "clutter_texture_split": args.asset_split,
                "clutter_texture_assets": [
                    {
                        "file": str(asset["path"]),
                        "catalog": asset.get("catalog"),
                        "source": asset.get("source"),
                        "commons_page": asset.get("commons_page"),
                        "license": asset.get("license"),
                        "license_url": asset.get("license_url"),
                        "artist": asset.get("artist"),
                        "attribution": asset.get("attribution"),
                        "source_sha256": asset.get("source_sha256"),
                    }
                    for asset in CLUTTER_TEXTURES
                ],
                "lighting_profiles": [profile[0] for profile in LIGHTING_PROFILES],
                "lighting_intensity_scale": args.lighting_scale,
                "gate_target_probability": args.gate_target_probability,
                "no_gates": args.no_gates,
                "counterfactual_gate_camera_policy": args.counterfactual_gate_camera_policy,
                "gate_view_sampling": {
                    "distance_m": [1.15, 2.0],
                    "maximum_horizontal_off_axis_degrees": 18,
                    "elevation_degrees": [-12, 18],
                    "purpose": "exclude far and extreme-angle corner-training views",
                },
                "camera_calibration": camera_calibration,
                "references": [
                    "https://newbeedrone.com/collections/all/products/newbeedrone-micro-race-gate-square",
                    "https://www.rotorama.com/product/newbeedrone-micro-race-gate-square",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    writer = rep.writers.get("BasicWriter")
    writer.initialize(
        backend=backend,
        rgb=True,
        distance_to_image_plane=True,
        semantic_segmentation=True,
        colorize_semantic_segmentation=False,
        semantic_types=["class"],
    )
    writer.attach(render_product)

    rng = np.random.default_rng(args.seed + args.start_index)
    manifest_path = args.output_dir / "poses.jsonl"
    records = []
    for local_index in range(args.frames):
        eye, target = sample_camera(rng)
        profile_name, dome_base, sun_base, temperature = LIGHTING_PROFILES[
            local_index % len(LIGHTING_PROFILES)
        ]
        dome.GetIntensityAttr().Set(float(args.lighting_scale * dome_base * rng.uniform(0.85, 1.15)))
        sun.GetIntensityAttr().Set(float(args.lighting_scale * sun_base * rng.uniform(0.85, 1.15)))
        sun.GetColorTemperatureAttr().Set(float(temperature + rng.uniform(-250.0, 250.0)))
        sun_rotate_x.Set(float(rng.uniform(25.0, 70.0)))
        sun_rotate_z.Set(float(rng.uniform(0.0, 360.0)))
        camera_matrix.Set(look_at_matrix(eye, target))
        rep.orchestrator.step(rt_subframes=args.rt_subframes, delta_time=0.0)
        records.append(
            json.dumps(
                {
                    "local_index": local_index,
                    "global_index": args.start_index + local_index,
                    "eye_m": eye,
                    "target_m": target,
                    "resolution": [args.width, args.height],
                    "depth_unit": "meter",
                    "semantic_classes": CLASS_NAMES,
                    "lighting_profile": profile_name,
                }
            )
        )
    rep.orchestrator.wait_until_complete()
    writer.detach()
    manifest_path.write_text("\n".join(records) + "\n")
    apply_sensor_model(args.output_dir, args.seed + args.start_index + 1771)


try:
    main()
except Exception:
    import traceback

    traceback.print_exc()
    app.close()
    raise
else:
    app.close()
