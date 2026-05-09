"""Small software graphics renderer with NumPy and linx backends."""

from .backends import BackendInfo, create_backend, list_backends
from .camera import Camera, look_at, perspective
from .mesh import Mesh
from .renderer import RenderResult, Renderer
from .scene import SceneHistory, SceneObject, SceneState
from .transforms import rotate_x, rotate_y, rotate_z, scale, translate

__all__ = [
    "BackendInfo",
    "Camera",
    "Mesh",
    "RenderResult",
    "Renderer",
    "SceneHistory",
    "SceneObject",
    "SceneState",
    "create_backend",
    "list_backends",
    "look_at",
    "perspective",
    "rotate_x",
    "rotate_y",
    "rotate_z",
    "scale",
    "translate",
]
