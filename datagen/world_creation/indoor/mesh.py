import colorsys
from collections import defaultdict
from typing import DefaultDict
from typing import Dict
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation
from shapely import affinity
from shapely.geometry import box as shapely_box
from trimesh import Trimesh
from trimesh import creation
from trimesh.visual import ColorVisuals

from common.errors import SwitchError
from datagen.world_creation.geometry import local_to_global_coords
from datagen.world_creation.indoor.blocks import BlocksByStory
from datagen.world_creation.indoor.blocks import CeilingBlock
from datagen.world_creation.indoor.blocks import FloorBlock
from datagen.world_creation.indoor.blocks import LadderBlock
from datagen.world_creation.indoor.blocks import LevelBlock
from datagen.world_creation.indoor.blocks import WallBlock
from datagen.world_creation.indoor.constants import DEFAULT_FLOOR_THICKNESS
from datagen.world_creation.indoor.constants import HIGH_POLY
from datagen.world_creation.indoor.constants import LOW_POLY
from datagen.world_creation.indoor.constants import CornerType
from datagen.world_creation.indoor.tiles import find_corners
from datagen.world_creation.indoor.tiles import find_exterior_wall_footprints
from datagen.world_creation.indoor.utils import get_evenly_spaced_centroids
from datagen.world_creation.types import Point3DNP
from datagen.world_creation.types import RGBATuple


def get_block_color(block: LevelBlock, aesthetics: "BuildingAestheticsConfig", rand: np.random.Generator) -> RGBATuple:
    if isinstance(block, WallBlock):
        if block.is_interior:
            color = aesthetics.interior_wall_color
            brightness_jitter = rand.normal(0.0, aesthetics.interior_wall_brightness_jitter)
            alpha = color[-1]
            h, s, v = colorsys.rgb_to_hsv(*color[:-1])
            v = float(np.clip(v + brightness_jitter, 0, 1))
            rgb = colorsys.hsv_to_rgb(h, s, v)
            rgb = [channel + rand.normal(0.0, aesthetics.interior_wall_color_jitter) for channel in rgb]
            color = (*rgb, alpha)
        else:
            color = aesthetics.exterior_color
    elif isinstance(block, FloorBlock):
        color = aesthetics.floor_color
    elif isinstance(block, CeilingBlock):
        color = aesthetics.ceiling_color
    elif isinstance(block, LadderBlock):
        color = aesthetics.ladder_color
    else:
        raise SwitchError(f"Unknown block type {type(block)}")
    return color


def homogeneous_transform_matrix(position: Point3DNP = np.array([0, 0, 0]), rotation: Optional[Rotation] = None):
    if rotation is None:
        rotation = np.eye(3)
    else:
        rotation = rotation.as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def make_color_visuals(mesh: Trimesh, rgba: RGBATuple):
    return ColorVisuals(mesh, face_colors=np.repeat([rgba], len(mesh.faces), axis=0))


class MeshData:
    def __init__(self):
        self.vertices = []
        self.faces = []
        self.face_normals = []
        self.face_colors = []
        self.index_offset = 0


def make_pretty_building_meshes(building: "Building", blocks_by_story: BlocksByStory) -> Dict[str, Trimesh]:
    rand = np.random.default_rng(building.id)
    corners_by_story = defaultdict(list)
    wall_footprints_by_story = defaultdict(list)
    exterior_wall_blocks_by_story = defaultdict(list)
    original_outlines_by_story = {}

    cornerless_building = building.rebuild_rotated(0)
    for story in cornerless_building.stories:
        outline = cornerless_building.get_footprint_outline(story.num)
        original_outlines_by_story[story.num] = outline

        new_footprint = story.footprint.copy()
        wall_footprints_by_story[story.num] = find_exterior_wall_footprints(new_footprint)
        corners = find_corners(new_footprint)
        corners_by_story[story.num] = corners
        for corner_position, corner_type, is_outside in corners:
            if is_outside:
                new_footprint[corner_position[0], corner_position[1]] = False
        story.footprint = new_footprint
        exterior_wall_blocks_by_story[story.num].extend(cornerless_building._generate_exterior_wall_blocks(story.num))

    mesh_datasets: DefaultDict[str, MeshData] = defaultdict(lambda: MeshData())

    x_offset = -building.width / 2
    z_offset = -building.length / 2
    tile_size = 1

    def add_mesh_data(parent_name, child_mesh):
        nonlocal mesh_datasets
        parent_mesh_data = mesh_datasets[parent_name]
        parent_mesh_data.vertices.extend(child_mesh.vertices)
        parent_mesh_data.faces.extend(child_mesh.faces + parent_mesh_data.index_offset)
        parent_mesh_data.face_normals.extend(child_mesh.face_normals)
        if child_mesh.visual:
            parent_mesh_data.face_colors.extend(child_mesh.visual.face_colors)
        else:
            parent_mesh_data.face_colors.extend([0, 0, 0] * len(child_mesh.faces))
        parent_mesh_data.index_offset += len(child_mesh.vertices)

    for story_num, blocks in blocks_by_story.items():
        story = building.stories[story_num]
        y_offset = building.get_story_y_offset(story_num)
        exterior_wall_blocks = exterior_wall_blocks_by_story[story.num]
        other_blocks = [block for block in blocks if not (isinstance(block, WallBlock) and not block.is_interior)]
        for block in [*exterior_wall_blocks, *other_blocks]:
            if not block.is_visual:
                continue

            translated_block = building.translate_block_to_building_space(
                block, story, building.aesthetics.block_downsize_epsilon
            )

            box = creation.box(translated_block.size)
            box = unnormalize(box)
            position = np.array([*translated_block.centroid])
            transform = homogeneous_transform_matrix(position)
            box.apply_transform(transform)
            box.visual = make_color_visuals(box, get_block_color(block, building.aesthetics, rand))
            add_mesh_data(HIGH_POLY, box)
            if (isinstance(block, WallBlock) and not block.is_interior) or isinstance(block, CeilingBlock):
                add_mesh_data(LOW_POLY, box)

        for story_link in story.story_links:
            if story_link.bottom_story_id != story.num:
                continue
            story_link_level_blocks = story_link.get_level_blocks(
                building.stories[story_link.bottom_story_id], building.stories[story_link.top_story_id]
            )
            for level_block in story_link_level_blocks:
                if not level_block.is_visual:
                    continue

                block_width, block_height, block_length = level_block.size
                block_x, block_y, block_z = level_block.centroid
                trimesh_box = creation.box((block_width, block_height, block_length))
                position = np.array(
                    [
                        x_offset + block_x,
                        building.get_story_y_offset(story_link.bottom_story_id) + DEFAULT_FLOOR_THICKNESS + block_y,
                        z_offset + block_z,
                    ]
                )
                rotation_transform = homogeneous_transform_matrix(rotation=level_block.rotation)
                trimesh_box.apply_transform(rotation_transform)
                translation_transform = homogeneous_transform_matrix(position)
                trimesh_box.apply_transform(translation_transform)
                trimesh_box = unnormalize(trimesh_box)
                trimesh_box.visual = make_color_visuals(
                    trimesh_box, get_block_color(level_block, building.aesthetics, rand)
                )
                add_mesh_data(HIGH_POLY, trimesh_box)

        outline = original_outlines_by_story[story_num]

        rail_height = building.aesthetics.rail_height
        rail_width = building.aesthetics.rail_thickness

        # constant: 1/2 of tile size
        buffer_width = 0.5

        min_x, min_y, max_x, max_y = outline.bounds
        width, length = max_x - min_x, max_y - min_y
        x_scaling_factor = width / (width + buffer_width * 2 + rail_width * building.aesthetics.rail_overhang_factor)
        z_scaling_factor = length / (length + buffer_width * 2 + rail_width * building.aesthetics.rail_overhang_factor)
        outline = outline.buffer(buffer_width, join_style=1, single_sided=True)
        outline = affinity.scale(outline, x_scaling_factor, z_scaling_factor)

        path = np.array([(c[0], y_offset + story.outer_height + rail_height, c[1]) for c in outline.exterior.coords])
        square = shapely_box(-rail_width / 2, 0, rail_width / 2, rail_height)
        border = creation.sweep_polygon(square, path)
        border = unnormalize(border)
        border.visual = make_color_visuals(border, building.aesthetics.trim_color)
        add_mesh_data(HIGH_POLY, border)

        offset = np.array([x_offset, 0, z_offset])

        crossbeam_size = building.aesthetics.crossbeam_size
        crossbeam_protrusion = building.aesthetics.crossbeam_protrusion
        crossbeam_centroid_y = y_offset + story.outer_height * building.aesthetics.crossbeam_y_proportion_of_height
        for wall_footprint in wall_footprints_by_story[story_num]:
            centroids = get_evenly_spaced_centroids(
                wall_footprint, crossbeam_size, building.aesthetics.crossbeam_min_gap, crossbeam_centroid_y
            )
            for centroid in centroids:
                centroid = local_to_global_coords(centroid, offset)
                transform = homogeneous_transform_matrix(position=centroid)
                if wall_footprint.is_vertical:
                    # Centroids are in wall middle, so adding +1 (wall thickness) to ensure they protrude
                    extents = (1 + crossbeam_protrusion, crossbeam_size, crossbeam_size)
                else:
                    extents = (crossbeam_size, crossbeam_size, 1 + crossbeam_protrusion)
                crossbeam = creation.box(extents, transform)
                crossbeam = unnormalize(crossbeam)
                crossbeam.visual = make_color_visuals(crossbeam, building.aesthetics.crossbeam_color)
                add_mesh_data(HIGH_POLY, crossbeam)

        for (z, x), corner_type, is_outside in corners_by_story[story_num]:
            if not is_outside:
                continue

            story = building.stories[story_num]

            corner_centroid = np.array(
                [x_offset + x + tile_size / 2, y_offset + (story.outer_height) / 2, z_offset + z + tile_size / 2]
            )
            rotation = Rotation.from_euler("x", 90, degrees=True)
            transform = homogeneous_transform_matrix(position=corner_centroid, rotation=rotation)
            round_corner_cylinder = creation.cylinder(tile_size / 2, story.outer_height, transform=transform)
            round_corner_cylinder = unnormalize(round_corner_cylinder)

            multiplier = 1 if corner_type in [CornerType.NE, CornerType.NW] else -1
            box_z = corner_centroid[2] + (multiplier * (tile_size / 4))
            box_centroid = np.array([corner_centroid[0], corner_centroid[1], box_z])
            transform = homogeneous_transform_matrix(position=box_centroid)
            horizontal_box = creation.box((tile_size, story.outer_height, tile_size / 2), transform)
            horizontal_box = unnormalize(horizontal_box)

            multiplier = 1 if corner_type in [CornerType.NW, CornerType.SW] else -1
            box_x = corner_centroid[0] + (multiplier * (tile_size / 4))
            box_centroid = np.array([box_x, corner_centroid[1], corner_centroid[2]])
            transform = homogeneous_transform_matrix(position=box_centroid)
            vertical_box = creation.box((tile_size / 2, story.outer_height, tile_size), transform)
            vertical_box = unnormalize(vertical_box)

            for component in [round_corner_cylinder, horizontal_box, vertical_box]:
                component.visual = make_color_visuals(component, building.aesthetics.exterior_color)
                add_mesh_data(HIGH_POLY, component)

    meshes = {}
    for name, mesh_data in mesh_datasets.items():
        mesh = Trimesh(
            mesh_data.vertices,
            mesh_data.faces,
            face_normals=mesh_data.face_normals,
            face_colors=mesh_data.face_colors,
            process=False,  # Don't merge identical vertices!
        )
        mesh.invert()  # Godot uses the opposite winding order of Trimesh
        meshes[name] = mesh
    return meshes


def unnormalize(mesh: Trimesh):
    """Re-create mesh such that none of its faces share vertices"""
    points_per_face = 3
    coords_per_point = 3
    face_count = len(mesh.faces)
    new_vertices = np.empty((face_count * points_per_face, coords_per_point), dtype=np.float32)
    for i, face in enumerate(mesh.faces):
        offset = i * points_per_face
        new_vertices[offset : offset + points_per_face, :] = mesh.vertices[face]
    new_faces = np.array(range(len(new_vertices))).reshape(-1, coords_per_point)
    return Trimesh(vertices=new_vertices, faces=new_faces, process=False)
