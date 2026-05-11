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
    object_ids: np.ndarray | None = None

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
        background_top: tuple[float, float, float] = (0.10, 0.11, 0.14),
        light_direction: tuple[float, float, float] = (-0.45, -0.7, -0.55),
        ambient: float = 0.18,
        diffuse: float = 0.82,
        specular: float = 0.28,
        shininess: float = 32.0,
        fog_strength: float = 0.42,
        normal_inverse_method: str = "schur",
        normal_inverse_min_block: int = 2,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        self.width = int(width)
        self.height = int(height)
        self.backend = create_backend(backend) if isinstance(backend, (str, type(None))) else backend
        self.background = np.asarray(background, dtype=np.float64)
        self.background_top = np.asarray(background_top, dtype=np.float64)
        self.light_direction = self.backend.normalize(light_direction)
        self.ambient = float(ambient)
        self.diffuse = float(diffuse)
        self.specular = float(specular)
        self.shininess = float(shininess)
        self.fog_strength = float(fog_strength)
        self.normal_inverse_method = normal_inverse_method
        self.normal_inverse_min_block = int(normal_inverse_min_block)

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
        items: Sequence[tuple[Mesh, np.ndarray] | tuple[Mesh, np.ndarray, int]],
        camera: Camera | None = None,
        backface_culling: bool = True,
    ) -> RenderResult:
        aspect = self.width / self.height
        camera = camera or Camera.look_at_perspective(aspect=aspect)

        parsed_items = self._parse_items(items)
        image = self._make_background()
        depth = np.full((self.height, self.width), np.inf, dtype=np.float64)
        object_ids = (
            np.full((self.height, self.width), -1, dtype=np.int64)
            if any(object_id is not None for _mesh, _model, object_id in parsed_items)
            else None
        )

        for mesh, model, object_id in parsed_items:
            self._draw_mesh(
                image,
                depth,
                object_ids,
                mesh,
                camera,
                np.asarray(model, dtype=np.float64),
                object_id,
                backface_culling,
            )

        return RenderResult(image=image, depth=depth, backend=self.backend.info, object_ids=object_ids)

    def _parse_items(
        self,
        items: Sequence[tuple[Mesh, np.ndarray] | tuple[Mesh, np.ndarray, int]],
    ) -> list[tuple[Mesh, np.ndarray, int | None]]:
        parsed: list[tuple[Mesh, np.ndarray, int | None]] = []
        for item in items:
            if len(item) == 2:
                mesh, model = item
                parsed.append((mesh, model, None))
            elif len(item) == 3:
                mesh, model, object_id = item
                parsed.append((mesh, model, int(object_id)))
            else:
                raise ValueError("render_many items must be (mesh, model) or (mesh, model, object_id)")
        return parsed

    def _make_background(self) -> np.ndarray:
        y = np.linspace(0.0, 1.0, self.height, dtype=np.float64).reshape(self.height, 1, 1)
        gradient = self.background_top.reshape(1, 1, 3) * (1.0 - y) + self.background.reshape(1, 1, 3) * y
        x_axis = np.linspace(-1.0, 1.0, self.width, dtype=np.float64).reshape(1, self.width, 1)
        y_axis = np.linspace(-1.0, 1.0, self.height, dtype=np.float64).reshape(self.height, 1, 1)
        vignette = 1.0 - 0.18 * np.clip(x_axis * x_axis + y_axis * y_axis, 0.0, 1.0)
        return np.clip(gradient * vignette, 0.0, 1.0)

    def _draw_mesh(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        object_ids: np.ndarray | None,
        mesh: Mesh,
        camera: Camera,
        model: np.ndarray,
        object_id: int | None,
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
        normal_matrix = self._normal_matrix(model)

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
            normal = self._face_normal(mesh.vertices[face], normal_matrix)
            color = self._shade_face(normal, world[face].mean(axis=0), mesh.face_colors[face_index], camera)
            color = self._apply_fog(color, screen[face])
            self._rasterize_triangle(
                image,
                depth,
                object_ids,
                screen[face],
                color,
                object_id,
                backface_culling,
            )

    def _normal_matrix(self, model: np.ndarray) -> np.ndarray:
        linear = np.ascontiguousarray(model[:3, :3], dtype=np.float64)
        return self.backend.inverse(
            linear,
            method=self.normal_inverse_method,
            min_block=self.normal_inverse_min_block,
        ).T

    def _face_normal(self, vertices: np.ndarray, normal_matrix: np.ndarray) -> np.ndarray:
        edge_a = vertices[1] - vertices[0]
        edge_b = vertices[2] - vertices[0]
        object_normal = np.cross(edge_a, edge_b)
        transformed = normal_matrix @ object_normal
        return _normalize(transformed)

    def _shade_face(
        self,
        normal: np.ndarray,
        center: np.ndarray,
        base_color: np.ndarray,
        camera: Camera,
    ) -> np.ndarray:
        if np.linalg.norm(normal) < 1e-12:
            return np.clip(base_color * self.ambient, 0.0, 1.0)
        diffuse = max(0.0, float(np.dot(normal, -self.light_direction)))
        if camera.eye is not None:
            view_dir = _normalize(camera.eye - center)
        else:
            view_dir = _normalize(np.array([0.0, 0.0, 1.0], dtype=np.float64) - center)
        half_dir = _normalize((-self.light_direction) + view_dir)
        spec_angle = max(0.0, float(np.dot(normal, half_dir)))
        specular = self.specular * (spec_angle ** self.shininess)
        rim = 0.14 * ((1.0 - max(0.0, float(np.dot(normal, view_dir)))) ** 2.0)

        lit = base_color * (self.ambient + self.diffuse * diffuse + rim)
        highlight = np.array([1.0, 0.96, 0.86], dtype=np.float64) * specular
        return np.clip(lit + highlight, 0.0, 1.0)

    def _apply_fog(self, color: np.ndarray, triangle: np.ndarray) -> np.ndarray:
        mean_depth = float(np.mean(triangle[:, 2]))
        fog = np.clip((mean_depth - 0.35) * self.fog_strength, 0.0, 0.55)
        return np.clip(color * (1.0 - fog) + self.background_top * fog, 0.0, 1.0)

    def _face_draw_order(self, faces: np.ndarray, screen: np.ndarray) -> np.ndarray:
        mean_depth = screen[faces, 2].mean(axis=1)
        return np.argsort(mean_depth)[::-1]

    def _inside_clip(self, ndc_triangle: np.ndarray) -> bool:
        return bool(np.all(ndc_triangle[:, 2] >= -1.0) and np.all(ndc_triangle[:, 2] <= 1.0))

    def _rasterize_triangle(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        object_ids: np.ndarray | None,
        triangle: np.ndarray,
        color: np.ndarray,
        object_id: int | None,
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
        if object_ids is not None and object_id is not None:
            object_ids[ymin : ymax + 1, xmin : xmax + 1][update] = object_id


def _edge(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0]))


def _edge_grid(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> np.ndarray:
    return (p[..., 0] - a[0]) * (b[1] - a[1]) - (p[..., 1] - a[1]) * (b[0] - a[0])


def _normalize(value: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(value)
    if norm < eps:
        return value * 0.0
    return value / norm
