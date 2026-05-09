"""Tiny triangle rasterizer with a toggleable matrix backend."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from .backends import BackendInfo, BaseBackend, create_backend
from .camera import Camera
from .io import save_image, to_uint8
from .mesh import Mesh
from .transforms import identity


@dataclass(frozen=True)
class RenderResult:
    image: np.ndarray
    depth: np.ndarray
    backend: BackendInfo

    def to_uint8(self) -> np.ndarray:
        return to_uint8(self.image)

    def save(self, path: str) -> None:
        save_image(path, self.image)


class Renderer:
    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        backend: str | BaseBackend | None = "auto",
        background: tuple[float, float, float] = (0.035, 0.04, 0.052),
        light_direction: tuple[float, float, float] = (-0.45, -0.7, -0.55),
        ambient: float = 0.22,
        diffuse: float = 0.82,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        self.width = int(width)
        self.height = int(height)
        self.backend = create_backend(backend) if isinstance(backend, (str, type(None))) else backend
        self.background = np.asarray(background, dtype=np.float64)
        self.light_direction = self.backend.normalize(light_direction)
        self.ambient = float(ambient)
        self.diffuse = float(diffuse)

    def render(
        self,
        mesh: Mesh,
        camera: Camera | None = None,
        model: np.ndarray | None = None,
        backface_culling: bool = True,
    ) -> RenderResult:
        return self.render_many(
            [(mesh, identity() if model is None else np.asarray(model, dtype=np.float64))],
            camera=camera,
            backface_culling=backface_culling,
        )

    def render_many(
        self,
        items: Sequence[tuple[Mesh, np.ndarray]],
        camera: Camera | None = None,
        backface_culling: bool = True,
    ) -> RenderResult:
        aspect = self.width / self.height
        camera = camera or Camera.look_at_perspective(aspect=aspect)

        image = np.tile(self.background.reshape(1, 1, 3), (self.height, self.width, 1))
        depth = np.full((self.height, self.width), np.inf, dtype=np.float64)

        for mesh, model in items:
            self._draw_mesh(
                image,
                depth,
                mesh,
                camera,
                np.asarray(model, dtype=np.float64),
                backface_culling,
            )

        return RenderResult(image=image, depth=depth, backend=self.backend.info)

    def _draw_mesh(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        mesh: Mesh,
        camera: Camera,
        model: np.ndarray,
        backface_culling: bool,
    ) -> None:
        vertices_h = np.concatenate(
            [mesh.vertices, np.ones((mesh.vertices.shape[0], 1), dtype=np.float64)],
            axis=1,
        )
        model_view = self.backend.matmul(camera.view, model)
        mvp = self.backend.matmul(camera.projection, model_view)
        clip = self.backend.matmul(vertices_h, mvp.T)
        world = self.backend.matmul(vertices_h, model.T)[:, :3]

        visible = clip[:, 3] > 1e-9
        ndc = np.zeros((clip.shape[0], 3), dtype=np.float64)
        ndc[visible] = clip[visible, :3] / clip[visible, 3:4]

        screen = np.empty_like(ndc)
        screen[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * (self.width - 1)
        screen[:, 1] = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * (self.height - 1)
        screen[:, 2] = ndc[:, 2] * 0.5 + 0.5

        face_order = self._face_draw_order(mesh.faces, screen)
        for face_index in face_order:
            face = mesh.faces[face_index]
            if not np.all(visible[face]):
                continue
            if not self._inside_clip(ndc[face]):
                continue
            shade = self._shade_face(world[face])
            color = np.clip(mesh.face_colors[face_index] * shade, 0.0, 1.0)
            self._rasterize_triangle(image, depth, screen[face], color, backface_culling)

    def _shade_face(self, vertices: np.ndarray) -> float:
        edge_a = vertices[1] - vertices[0]
        edge_b = vertices[2] - vertices[0]
        normal = np.cross(edge_a, edge_b)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            return self.ambient
        normal /= norm
        diffuse = max(0.0, float(np.dot(normal, -self.light_direction)))
        return min(1.0, self.ambient + self.diffuse * diffuse)

    def _face_draw_order(self, faces: np.ndarray, screen: np.ndarray) -> np.ndarray:
        mean_depth = screen[faces, 2].mean(axis=1)
        return np.argsort(mean_depth)[::-1]

    def _inside_clip(self, ndc_triangle: np.ndarray) -> bool:
        return bool(np.all(ndc_triangle[:, 2] >= -1.0) and np.all(ndc_triangle[:, 2] <= 1.0))

    def _rasterize_triangle(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        triangle: np.ndarray,
        color: np.ndarray,
        backface_culling: bool,
    ) -> None:
        xy = triangle[:, :2]
        z = triangle[:, 2]
        area = _edge(xy[0], xy[1], xy[2])
        if abs(area) < 1e-9:
            return
        if backface_culling and area < 0.0:
            return

        xmin = max(0, int(np.floor(np.min(xy[:, 0]))))
        xmax = min(self.width - 1, int(np.ceil(np.max(xy[:, 0]))))
        ymin = max(0, int(np.floor(np.min(xy[:, 1]))))
        ymax = min(self.height - 1, int(np.ceil(np.max(xy[:, 1]))))
        if xmin > xmax or ymin > ymax:
            return

        ys, xs = np.mgrid[ymin : ymax + 1, xmin : xmax + 1]
        p = np.stack([xs + 0.5, ys + 0.5], axis=-1)
        w0 = _edge_grid(xy[1], xy[2], p) / area
        w1 = _edge_grid(xy[2], xy[0], p) / area
        w2 = _edge_grid(xy[0], xy[1], p) / area
        mask = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not np.any(mask):
            return

        tri_depth = w0 * z[0] + w1 * z[1] + w2 * z[2]
        patch_depth = depth[ymin : ymax + 1, xmin : xmax + 1]
        update = mask & (tri_depth < patch_depth)
        if not np.any(update):
            return

        patch_image = image[ymin : ymax + 1, xmin : xmax + 1]
        patch_depth[update] = tri_depth[update]
        patch_image[update] = color


def _edge(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0]))


def _edge_grid(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> np.ndarray:
    return (p[..., 0] - a[0]) * (b[1] - a[1]) - (p[..., 1] - a[1]) * (b[0] - a[0])
