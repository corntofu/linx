"""Small editable scene model used by the GUI."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .mesh import Mesh
from .transforms import rotate_x, rotate_y, rotate_z, scale, translate


@dataclass(frozen=True)
class SceneObject:
    id: int
    kind: str
    name: str
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0
    size: float = 1.0

    def mesh(self) -> Mesh:
        if self.kind == "cube":
            return Mesh.cube(size=1.4)
        if self.kind == "pyramid":
            return Mesh.pyramid(size=1.5, height=1.7)
        if self.kind == "sphere":
            return Mesh.uv_sphere()
        if self.kind == "torus":
            return Mesh.torus()
        raise ValueError(f"unknown shape kind {self.kind!r}")

    def model_matrix(self) -> np.ndarray:
        return (
            translate(self.x, self.y, self.z)
            @ rotate_y(math.radians(self.ry))
            @ rotate_x(math.radians(self.rx))
            @ rotate_z(math.radians(self.rz))
            @ scale(self.size)
        )

    def rotated(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> "SceneObject":
        return replace(self, rx=self.rx + dx, ry=self.ry + dy, rz=self.rz + dz)

    def moved(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> "SceneObject":
        return replace(self, x=self.x + dx, y=self.y + dy, z=self.z + dz)


@dataclass(frozen=True)
class SceneState:
    objects: tuple[SceneObject, ...] = ()
    selected_id: int | None = None
    next_id: int = 1

    def selected(self) -> SceneObject | None:
        for obj in self.objects:
            if obj.id == self.selected_id:
                return obj
        return None


class SceneHistory:
    """Immutable scene snapshots with undo/redo support."""

    def __init__(self, initial: SceneState | None = None) -> None:
        self._undo: list[SceneState] = []
        self._redo: list[SceneState] = []
        self.state = initial or SceneState()

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def set_state(self, state: SceneState, record: bool = True) -> SceneState:
        if record:
            self._undo.append(self.state)
            self._redo.clear()
        self.state = state
        return self.state

    def add_shape(self, kind: str) -> SceneState:
        if kind not in {"cube", "pyramid", "sphere", "torus"}:
            raise ValueError("kind must be 'cube', 'pyramid', 'sphere', or 'torus'")
        shape_id = self.state.next_id
        offset = ((shape_id - 1) % 5 - 2) * 0.72
        obj = SceneObject(
            id=shape_id,
            kind=kind,
            name=f"{kind.title()} {shape_id}",
            x=offset,
            ry=25.0 + shape_id * 12.0,
            size=0.92,
        )
        return self.set_state(
            SceneState(
                objects=self.state.objects + (obj,),
                selected_id=shape_id,
                next_id=shape_id + 1,
            )
        )

    def remove_selected(self) -> SceneState:
        selected_id = self.state.selected_id
        if selected_id is None:
            return self.state
        remaining = tuple(obj for obj in self.state.objects if obj.id != selected_id)
        next_selected = remaining[-1].id if remaining else None
        return self.set_state(
            SceneState(
                objects=remaining,
                selected_id=next_selected,
                next_id=self.state.next_id,
            )
        )

    def select(self, object_id: int | None) -> SceneState:
        if object_id is not None and all(obj.id != object_id for obj in self.state.objects):
            return self.state
        self.state = SceneState(
            objects=self.state.objects,
            selected_id=object_id,
            next_id=self.state.next_id,
        )
        return self.state

    def rotate_selected(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> SceneState:
        selected_id = self.state.selected_id
        if selected_id is None:
            return self.state
        updated = tuple(
            obj.rotated(dx=dx, dy=dy, dz=dz) if obj.id == selected_id else obj
            for obj in self.state.objects
        )
        return self.set_state(
            SceneState(
                objects=updated,
                selected_id=selected_id,
                next_id=self.state.next_id,
            )
        )

    def move_selected(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
        record: bool = True,
    ) -> SceneState:
        selected_id = self.state.selected_id
        if selected_id is None:
            return self.state
        updated = tuple(
            obj.moved(dx=dx, dy=dy, dz=dz) if obj.id == selected_id else obj
            for obj in self.state.objects
        )
        return self.set_state(
            SceneState(
                objects=updated,
                selected_id=selected_id,
                next_id=self.state.next_id,
            ),
            record=record,
        )

    def clear(self) -> SceneState:
        if not self.state.objects:
            return self.state
        return self.set_state(SceneState(next_id=self.state.next_id))

    def undo(self) -> SceneState:
        if not self._undo:
            return self.state
        self._redo.append(self.state)
        self.state = self._undo.pop()
        return self.state

    def redo(self) -> SceneState:
        if not self._redo:
            return self.state
        self._undo.append(self.state)
        self.state = self._redo.pop()
        return self.state

    def render_items(self, include_ids: bool = False, include_ground: bool = False):
        items = []
        if include_ground:
            items.append((Mesh.ground(), np.eye(4, dtype=np.float64)))
        if include_ids:
            items.extend((obj.mesh(), obj.model_matrix(), obj.id) for obj in self.state.objects)
        else:
            items.extend((obj.mesh(), obj.model_matrix()) for obj in self.state.objects)
        return items
