from __future__ import annotations

import unittest

import numpy as np

from lingraphics.backends import create_backend
from lingraphics.camera import Camera
from lingraphics.mesh import Mesh
from lingraphics.renderer import Renderer
from lingraphics.scene import SceneHistory
from lingraphics.transforms import rotate_x, rotate_y


class RendererTests(unittest.TestCase):
    def test_numpy_backend_renders_non_empty_image(self) -> None:
        renderer = Renderer(width=96, height=72, backend="numpy")
        camera = Camera.look_at_perspective(aspect=96 / 72)
        result = renderer.render(Mesh.cube(), camera=camera, model=rotate_y(0.55) @ rotate_x(0.3))

        self.assertEqual(result.image.shape, (72, 96, 3))
        self.assertEqual(result.depth.shape, (72, 96))
        self.assertTrue(np.isfinite(result.depth).any())
        self.assertGreater(float(result.image.max()), float(renderer.background.max()))

    def test_backend_toggle_numpy(self) -> None:
        backend = create_backend("numpy")
        self.assertEqual(backend.name, "numpy")
        product = backend.matmul(np.eye(4), np.ones((4, 4)))
        np.testing.assert_allclose(product, np.ones((4, 4)))

    def test_backend_toggle_linx_when_available(self) -> None:
        try:
            backend = create_backend("linx")
        except RuntimeError:
            self.skipTest("linx extension is not available")
        self.assertEqual(backend.name, "linx")
        product = backend.matmul(np.eye(4), np.ones((4, 4)))
        np.testing.assert_allclose(product, np.ones((4, 4)))

    def test_scene_history_add_delete_undo_redo(self) -> None:
        history = SceneHistory()
        history.add_shape("cube")
        history.add_shape("pyramid")

        self.assertEqual(len(history.state.objects), 2)
        self.assertEqual(history.state.selected().kind, "pyramid")

        history.remove_selected()
        self.assertEqual(len(history.state.objects), 1)
        self.assertEqual(history.state.selected().kind, "cube")

        history.undo()
        self.assertEqual(len(history.state.objects), 2)
        self.assertEqual(history.state.selected().kind, "pyramid")

        history.redo()
        self.assertEqual(len(history.state.objects), 1)

    def test_scene_history_move_selected_undo(self) -> None:
        history = SceneHistory()
        history.add_shape("cube")
        before = history.state.selected()

        history.move_selected(dx=0.5, dy=-0.25, dz=0.1)
        after = history.state.selected()
        self.assertAlmostEqual(after.x, before.x + 0.5)
        self.assertAlmostEqual(after.y, before.y - 0.25)
        self.assertAlmostEqual(after.z, before.z + 0.1)

        history.undo()
        restored = history.state.selected()
        self.assertAlmostEqual(restored.x, before.x)
        self.assertAlmostEqual(restored.y, before.y)
        self.assertAlmostEqual(restored.z, before.z)

    def test_render_many_renders_scene(self) -> None:
        history = SceneHistory()
        history.add_shape("cube")
        history.add_shape("pyramid")
        renderer = Renderer(width=96, height=72, backend="numpy")
        result = renderer.render_many(history.render_items(include_ids=True))

        self.assertEqual(result.image.shape, (72, 96, 3))
        self.assertIsNotNone(result.object_ids)
        self.assertTrue((result.object_ids >= 0).any())
        self.assertTrue(np.isfinite(result.depth).any())


if __name__ == "__main__":
    unittest.main()
