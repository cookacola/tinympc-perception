#!/usr/bin/env python3
"""Generate one resumable shard of aligned racetrack RGB/depth/semantics."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
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
    return parser.parse_args()


args = parse_args()
args.output_dir = args.output_dir.expanduser().resolve()
args.camera_calibration = args.camera_calibration.expanduser().resolve()
if not 0.0 <= args.gate_target_probability <= 1.0:
    raise ValueError("gate target probability must be in [0, 1]")
camera_calibration = json.loads(args.camera_calibration.read_text())
sys.argv = [sys.argv[0]]
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp

app = SimulationApp(
    launch_config={
        "headless": True,
        "width": args.width,
        "height": args.height,
        "renderer": "RayTracedLighting",
    }
)

import carb.settings
import omni.kit.app
import omni.replicator.core as rep
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt

extension_manager = omni.kit.app.get_app().get_extension_manager()
if not extension_manager.set_extension_enabled_immediate(
    "isaacsim.sensors.experimental.rtx", True
):
    raise RuntimeError("failed to enable RTX sensor/lens-distortion schemas")

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
    rep.functional.create.dome_light(intensity=120, parent="/World", name="DomeLight")
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(900.0)
    sun.CreateAngleAttr(1.0)

    # Lab floor plus one fixed, approximately 4 m x 4 m wooden-tile course.
    create_box(
        "LabFloor", (0, 0, -0.08), (8, 8, 0.12), "lab_clutter",
        parent="/World", color=(0.35, 0.37, 0.39),
    )
    for row in range(8):
        for column in range(8):
            x = -1.75 + column * 0.5
            y = -1.75 + row * 0.5
            wood = (0.50, 0.24, 0.08) if (row + column) % 2 else (0.68, 0.36, 0.12)
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

    # Exactly two obstacles.
    create_box(
        "Obstacle_0", (-0.55, 0.65, 0.30), (0.55, 0.55, 0.60),
        "obstacle", color=(0.12, 0.28, 0.65),
    )
    create_box(
        "Obstacle_1", (0.60, -0.65, 0.38), (0.50, 0.75, 0.76),
        "obstacle", color=(0.68, 0.14, 0.10),
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
    for gate_index, (x_pos, y_center) in enumerate(((-1.30, -0.45), (1.30, 0.45))):
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
        ("Computer_W", (-2.70, 0.0, 1.15), (0.12, 0.65, 0.48), (0.03, 0.04, 0.05)),
        ("Computer_E", (2.70, 0.1, 1.15), (0.12, 0.65, 0.48), (0.03, 0.04, 0.05)),
        ("Shelf_Back", (0.0, 3.10, 1.10), (3.0, 0.45, 2.2), (0.25, 0.27, 0.30)),
        ("Shelf_Side", (-3.25, -2.7, 1.00), (0.60, 1.5, 2.0), (0.25, 0.27, 0.30)),
    ]:
        create_box(
            name, position, scale, "lab_clutter", parent=clutter_parent, color=color
        )

    camera = rep.functional.create.camera(
        position=(0, -12, 4), look_at=(0, 0, 0), parent="/World", name="Camera"
    )
    if not camera.ApplyAPI("OmniLensDistortionOpenCvPinholeAPI"):
        raise RuntimeError("OpenCV pinhole lens schema is unavailable")
    camera.GetAttribute("omni:lensdistortion:model").Set("opencvPinhole")
    intrinsic = camera_calibration["camera_matrix"]
    distortion = camera_calibration["distortion_coefficients"]
    pinhole_values = {
        "fx": intrinsic[0][0],
        "fy": intrinsic[1][1],
        "cx": intrinsic[0][2],
        "cy": intrinsic[1][2],
        "k1": distortion[0],
        "k2": distortion[1],
        "p1": distortion[2],
        "p2": distortion[3],
        "k3": distortion[4],
    }
    camera.GetAttribute(
        "omni:lensdistortion:opencvPinhole:imageSize"
    ).Set(Gf.Vec2i(args.width, args.height))
    for attribute, value in pinhole_values.items():
        usd_attribute = camera.GetAttribute(
            f"omni:lensdistortion:opencvPinhole:{attribute}"
        )
        if not usd_attribute:
            raise RuntimeError(f"missing OpenCV pinhole attribute: {attribute}")
        usd_attribute.Set(float(value))
    camera_xform = UsdGeom.Xformable(camera)
    camera_xform.ClearXformOpOrder()
    camera_matrix = camera_xform.MakeMatrixXform()
    render_product = rep.create.render_product(
        camera, (args.width, args.height), name="CourseRenderProduct"
    )
    return camera_matrix, render_product


def sample_camera(rng):
    # Aim primarily at course features so the physically small micro gates and
    # both obstacles are well represented while retaining broad pose diversity.
    points = np.asarray(
        [(-1.30, -0.45, 0.55), (1.30, 0.45, 0.55),
         (-0.55, 0.65, 0.30), (0.60, -0.65, 0.38)]
    )
    if rng.random() < args.gate_target_probability:
        point_index = int(rng.integers(0, 2))
    else:
        point_index = int(rng.integers(2, len(points)))
    target = points[point_index].copy()
    target += rng.normal(
        (0, 0, 0), (0.06, 0.06, 0.05) if point_index < 2 else (0.20, 0.20, 0.12)
    )
    if point_index < 2:
        # The gate plane is Y-Z: approach it primarily along X so the square
        # opening, rather than its 25 mm edge, is visible. Approach each gate
        # from the course interior; approaching from the outside would clamp
        # the eye against the course boundary and put the fabric frame almost
        # on the camera.
        # Keep the 0.555 m label square roughly 51--88 px wide at fx=183 px:
        # large enough for corner regression without near-field cropping.
        distance = float(rng.uniform(1.15, 2.0))
        inward_azimuth = math.pi if point_index == 0 else 0.0
        azimuth_offset = float(rng.uniform(-math.radians(18), math.radians(18)))
        azimuth = inward_azimuth + azimuth_offset
    else:
        # Keep the camera outside the two obstacle volumes.
        distance = float(rng.uniform(1.8, 2.8))
        azimuth = float(rng.uniform(-math.pi, math.pi))
    x = float(np.clip(target[0] - distance * math.cos(azimuth), -1.85, 1.85))
    y = float(np.clip(target[1] - distance * math.sin(azimuth), -1.85, 1.85))
    if point_index < 2:
        target[2] = float(np.clip(target[2], 0.42, 0.62))
        elevation = float(rng.uniform(math.radians(-12), math.radians(18)))
        z = float(target[2] + distance * math.tan(elevation))
    else:
        target[2] = float(np.clip(target[2], 0.18, 0.62))
        z = float(np.clip(target[2] + rng.uniform(0.25, 1.00), 0.48, 1.50))
    return (x, y, z), tuple(float(v) for v in target)


def main():
    camera_matrix, render_product = build_scene()
    backend = rep.backends.get("DiskBackend")
    backend.initialize(output_dir=str(args.output_dir))
    (args.output_dir / "scene_metadata.json").write_text(
        json.dumps(
            {
                "course_count": 1,
                "course_extent_m": [4.0, 4.0],
                "floor": "alternating wooden tiles",
                "obstacle_count": 2,
                "gate_count": 2,
                "gate_model": "NewBeeDrone Micro Race Gate - Square",
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
                "randomized": ["camera_position", "camera_look_at"],
                "gate_target_probability": args.gate_target_probability,
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
                }
            )
        )
    rep.orchestrator.wait_until_complete()
    writer.detach()
    manifest_path.write_text("\n".join(records) + "\n")


try:
    main()
except Exception:
    import traceback

    traceback.print_exc()
    app.close()
    raise
else:
    app.close()
