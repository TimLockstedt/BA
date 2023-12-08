import warnings
from functools import cache

import numba
import numpy as np
import tqdm
import vispy.io as io
from vispy import scene
from vispy.visuals.filters import ShadingFilter, WireframeFilter
from vispy.visuals.transforms import STTransform

from _sh4 import spherical_harmonics


@numba.njit(cache=True)
def rgb_color(phi, incl):
    incl = np.abs(incl)
    return np.abs(
        np.array(
            [np.cos(phi) * np.cos(incl), np.sin(phi) * np.cos(incl), np.sin(incl)],
            dtype=np.float32,
        )
    )


@numba.njit(cache=True)
def _vertex_color_rgb(vertices):
    color_vertices = np.empty((len(vertices), 3), dtype=np.float32)
    for i, vertex in enumerate(vertices):
        p = vertex
        phi = np.arctan2(p[1], p[0])
        l = np.linalg.norm(p)
        if l > 0:
            incl = np.arcsin(p[2] / np.linalg.norm(p))
        else:
            incl = 0
        color_vertices[i] = rgb_color(phi, incl)
    return color_vertices


@numba.njit(cache=True)
def _radius(
    p: np.array,
    coefficients: np.array,
    asymetric: bool,
    l_max: int,
):
    phi = np.arctan2(p[1], p[0])
    theta = np.arccos(p[2] / np.linalg.norm(p))

    costheta = np.cos(theta)
    sintheta = np.sin(theta)

    count = 0
    radius = 0
    l_step = 1 if asymetric else 2

    for band in range(0, l_max, l_step):
        for order in range(-band, band + 1):
            radius += (
                spherical_harmonics(band, order, costheta, sintheta, phi)
                * coefficients[count]
            )
            count += 1
            if count >= len(coefficients):
                break
        if count >= len(coefficients):
            break

    return radius


def get_l_max_sym(coeff: np.array) -> int:
    return int(1 / 2 * (np.sqrt(8 * coeff + 1) - 3))


def get_l_max_asym(coeff: np.array) -> int:
    return int(np.sqrt(coeff.size) - 1)


@cache
def get_num_asym_coeff(x: int) -> int:
    return (x + 1) ** 2


@cache
def get_num_sym_coeff(bands: int) -> int:
    return ((bands // 2) + 1) * (2 * (bands // 2) + 1)


@cache
def get_list_num_asym_coeff(bands: int) -> int:
    return [get_num_asym_coeff(x) for x in range(1, bands + 1)]


@cache
def get_list_num_sym_coeff(bands: int) -> int:
    return [get_num_sym_coeff(x) for x in range(1, bands + 1)]


def is_sym(coeff: np.array) -> bool:
    l_asym = get_list_num_asym_coeff(42)
    l_sym = get_list_num_sym_coeff(42)

    if coeff.size not in l_asym and coeff.size not in l_sym:
        raise ValueError(f"invalid coeff number: {coeff.size}")

    return coeff.size in l_sym


def create_odf(
    odf_coeff,
    l_max,
    translate=[0, 0, 0],
    radius=1,
    n_sphere=64,
    shading_filter=True,
    draw_mesh=False,
    negative_values=True,
):
    """ico seems to have problems with culling, use latitude -> -shadow- problem"""

    assert n_sphere > 1

    sphere = scene.visuals.Sphere(
        radius=1,
        method="latitude",
        cols=n_sphere,
        rows=n_sphere,
        shading=None,
        edge_color=None,
    )
    vertices = sphere.mesh.mesh_data.get_vertices()

    asymetric = is_sym(odf_coeff) is False

    negs = np.zeros(len(vertices), dtype=bool)
    for i, v in enumerate(vertices):
        r = _radius(v, odf_coeff, asymetric, l_max=l_max)
        if not negative_values:
            if r < 0:
                negs[i] = True
                r = 0

        v *= r
        v *= radius

    # coloring
    colors = _vertex_color_rgb(vertices)
    # if np.any(negs):
    #     print("negs:", np.sum(negs))
    colors[negs] = 1

    sphere.mesh.mesh_data.set_vertex_colors(colors)

    # Shading
    if draw_mesh is True:
        wireframe_filter = WireframeFilter(width=0.25)
        sphere.mesh.attach(wireframe_filter)

    if shading_filter is True:
        shading_filter = ShadingFilter(
            shading="flat",  # 'smooth', but no visible difference
            shininess=25,
            light_dir=(0, 0, -10),
            ambient_light=1.0,  # color
            ambient_coefficient=0.5,  # reflection
            specular_light=1.0,  # color
            specular_coefficient=(1, 1, 1, 0.1),  # reflection
            diffuse_coefficient=0.7,  # reflection
            enabled=True,
        )
        sphere.mesh.attach(
            shading_filter
        )  # TODO: attach maybe not corret?, should overwrite?

    # TODO: I guess, not every mesh should have an individual shading_filter

    sphere.transform = STTransform(translate=translate)

    return sphere


def render_scene(
    odf_coeffs,
    output=None,
    l_max = 12,
    n_sphere=64,
    size_per_odf=256,  # pixelsize
    scale=1,
    downscale=1,
    negative_values=False,
    bgcolor="transparent",
    draw_axis=False,
):
    if odf_coeffs.ndim == 1:
        odf_coeffs = odf_coeffs[np.newaxis, np.newaxis, :]
    elif odf_coeffs.ndim == 2:
        odf_coeffs = odf_coeffs[np.newaxis, :]
    elif odf_coeffs.ndim > 3:
        raise ValueError("odf_coeffs.ndim > 3")

    asymetric = is_sym(odf_coeffs[0, 0, :]) is False
    print("asymetric:", asymetric)

    # canvas
    size = (odf_coeffs.shape[0] * size_per_odf, odf_coeffs.shape[1] * size_per_odf)
    canvas = scene.SceneCanvas(bgcolor=bgcolor, size=size)
    view = canvas.central_widget.add_view()
    seperation = 2  # distance between odfs

    view.camera = "panzoom"
    view.camera.rect = (
        -seperation / 2,
        -seperation / 2,
        odf_coeffs.shape[0] * seperation,
        odf_coeffs.shape[1] * seperation,
    )

    assert view.camera.fov == 0

    for i, j in tqdm.tqdm(
        np.ndindex(odf_coeffs.shape[:2]),
        desc="rendering odfs",
        total=np.prod(odf_coeffs.shape[:2]),
    ):
        if np.sum(np.abs(odf_coeffs[i, j])) == 0:
            continue
        elif np.any(np.isnan(odf_coeffs[i, j])):
            continue
        view.add(
            create_odf(
                odf_coeffs[i, j, :],
                l_max=l_max,
                translate=[2 * i, 2 * j, 0],
                radius=scale,
                n_sphere=n_sphere,
                negative_values=negative_values,
            )
        )

    # Axes
    if draw_axis is True:
        if bgcolor == "transparent":
            warnings.warn("lines on transparent background are not visible")
        # TODO: vispy.scene.visuals.Axis?
        line = scene.visuals.Line(pos=[[0, 0, 0], [1, 0, 0]], color=[228/256, 26/256, 28/256,1], width=1)
        view.add(line)
        line = scene.visuals.Line(pos=[[0, 0, 0], [0, 1, 0]], color=[74/256, 175/256, 74/256,1], width=1)
        view.add(line)
        line = scene.visuals.Line(pos=[[0, 0, 0], [0, 0, 1]], color=[55/256, 126/256, 184/256,1], width=1)
        view.add(line)

        line = scene.visuals.Line(
            pos=[
                [-seperation / 2, -seperation / 2, 0],
                [-seperation / 2, +seperation / 2, 0],
                [+seperation / 2, +seperation / 2, 0],
                [+seperation / 2, -seperation / 2, 0],
                [-seperation / 2, -seperation / 2, 0],
            ],
            color=[1, 1, 1],
        )
        view.add(line)

    # render
    img = canvas.render()

    if output is not None:
        io.write_png(output, img)

    return img
