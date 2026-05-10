"""Tkinter GUI for creating and editing lingraphics scenes."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

from .benchmark import RenderTiming, SchurTiming, benchmark_backends, benchmark_schur_inverse, parse_sizes
from .camera import Camera
from .io import to_uint8
from .renderer import Renderer
from .scene import SceneHistory


class LinGraphicsApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("lingraphics")
        self.geometry("980x680")
        self.minsize(820, 560)

        self.backend_var = tk.StringVar(value="auto")
        self.shape_var = tk.StringVar(value="cube")
        self.schur_sizes_var = tk.StringVar(value="64,128,256")
        self.status_var = tk.StringVar(value="")
        self.history = SceneHistory()
        self.renderer: Renderer | None = None
        self._photo: tk.PhotoImage | None = None
        self._object_ids: np.ndarray | None = None
        self._drag_start: tuple[int, int] | None = None
        self._drag_base_state = None
        self._drag_has_motion = False

        self._build_ui()
        self.history.add_shape("cube")
        self.history.add_shape("pyramid")
        self._sync_ui()
        self._render()

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(root, width=230)
        sidebar.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        sidebar.grid_propagate(False)

        ttk.Label(sidebar, text="Backend").pack(anchor="w")
        backend = ttk.Combobox(
            sidebar,
            textvariable=self.backend_var,
            values=("auto", "numpy", "linx"),
            state="readonly",
        )
        backend.pack(fill=tk.X, pady=(4, 12))
        backend.bind("<<ComboboxSelected>>", lambda _event: self._render())

        ttk.Label(sidebar, text="Shape").pack(anchor="w")
        shape_row = ttk.Frame(sidebar)
        shape_row.pack(fill=tk.X, pady=(4, 8))
        ttk.Combobox(
            shape_row,
            textvariable=self.shape_var,
            values=("cube", "pyramid", "sphere", "torus"),
            state="readonly",
            width=12,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(shape_row, text="Add", command=self._add_shape).pack(side=tk.LEFT, padx=(6, 0))

        actions = ttk.Frame(sidebar)
        actions.pack(fill=tk.X, pady=(0, 10))
        self.delete_button = ttk.Button(actions, text="Delete", command=self._delete_selected)
        self.delete_button.grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=(0, 6))
        self.undo_button = ttk.Button(actions, text="Undo", command=self._undo)
        self.undo_button.grid(row=0, column=1, sticky="ew", padx=(4, 0), pady=(0, 6))
        self.redo_button = ttk.Button(actions, text="Redo", command=self._redo)
        self.redo_button.grid(row=1, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(actions, text="Clear", command=self._clear_scene).grid(row=1, column=1, sticky="ew", padx=(4, 0))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)

        ttk.Button(sidebar, text="Benchmark Render", command=self._benchmark).pack(fill=tk.X, pady=(0, 8))
        ttk.Label(sidebar, text="Schur sizes").pack(anchor="w")
        ttk.Entry(sidebar, textvariable=self.schur_sizes_var).pack(fill=tk.X, pady=(4, 6))
        ttk.Button(sidebar, text="Benchmark Schur", command=self._benchmark_schur).pack(fill=tk.X, pady=(0, 10))

        rotate_box = ttk.LabelFrame(sidebar, text="Rotate Selected", padding=8)
        rotate_box.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(rotate_box, text="X", command=lambda: self._rotate(dx=15)).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(rotate_box, text="Y", command=lambda: self._rotate(dy=15)).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(rotate_box, text="Z", command=lambda: self._rotate(dz=15)).grid(row=0, column=2, sticky="ew", padx=(4, 0))
        rotate_box.columnconfigure(0, weight=1)
        rotate_box.columnconfigure(1, weight=1)
        rotate_box.columnconfigure(2, weight=1)

        move_box = ttk.LabelFrame(sidebar, text="Move Selected", padding=8)
        move_box.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(move_box, text="Up", command=lambda: self._move(dy=0.18)).grid(row=0, column=1, sticky="ew", pady=(0, 4))
        ttk.Button(move_box, text="Left", command=lambda: self._move(dx=-0.18)).grid(row=1, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(move_box, text="Right", command=lambda: self._move(dx=0.18)).grid(row=1, column=2, sticky="ew", padx=(4, 0))
        ttk.Button(move_box, text="Down", command=lambda: self._move(dy=-0.18)).grid(row=2, column=1, sticky="ew", pady=(4, 0))
        ttk.Button(move_box, text="Near", command=lambda: self._move(dz=0.18)).grid(row=3, column=0, sticky="ew", padx=(0, 4), pady=(8, 0))
        ttk.Button(move_box, text="Far", command=lambda: self._move(dz=-0.18)).grid(row=3, column=2, sticky="ew", padx=(4, 0), pady=(8, 0))
        move_box.columnconfigure(0, weight=1)
        move_box.columnconfigure(1, weight=1)
        move_box.columnconfigure(2, weight=1)

        ttk.Label(sidebar, text="Objects").pack(anchor="w")
        self.object_list = tk.Listbox(sidebar, height=14, exportselection=False)
        self.object_list.pack(fill=tk.BOTH, expand=True, pady=(4, 8))
        self.object_list.bind("<<ListboxSelect>>", self._select_from_list)

        ttk.Label(sidebar, textvariable=self.status_var, wraplength=210, foreground="#4b5563").pack(fill=tk.X)

        viewport = ttk.Frame(root)
        viewport.grid(row=0, column=1, sticky="nsew")
        viewport.rowconfigure(0, weight=1)
        viewport.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(viewport, width=720, height=540, bg="#090b10", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self._render_later())
        self.canvas.bind("<Button-1>", self._canvas_press)
        self.canvas.bind("<B1-Motion>", self._canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._canvas_release)

        results_frame = ttk.LabelFrame(viewport, text="Benchmark Results", padding=6)
        results_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        results_frame.columnconfigure(0, weight=1)
        self.results_table = ttk.Treeview(results_frame, show="headings", height=6)
        self.results_table.grid(row=0, column=0, sticky="ew")
        y_scroll = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_table.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = ttk.Scrollbar(results_frame, orient=tk.HORIZONTAL, command=self.results_table.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.results_table.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self._set_results_table(
            [("status", "Status", 760)],
            [("Run Benchmark Render or Benchmark Schur to populate this table.",)],
        )

        self.bind("<Command-z>", lambda _event: self._undo())
        self.bind("<Control-z>", lambda _event: self._undo())
        self.bind("<BackSpace>", lambda _event: self._delete_selected())
        self.bind("<Delete>", lambda _event: self._delete_selected())
        self.bind("<Left>", lambda _event: self._move(dx=-0.18))
        self.bind("<Right>", lambda _event: self._move(dx=0.18))
        self.bind("<Up>", lambda _event: self._move(dy=0.18))
        self.bind("<Down>", lambda _event: self._move(dy=-0.18))
        self.bind("w", lambda _event: self._move(dz=0.18))
        self.bind("s", lambda _event: self._move(dz=-0.18))

    def _add_shape(self) -> None:
        self.history.add_shape(self.shape_var.get())
        self._sync_ui()
        self._render()

    def _delete_selected(self) -> None:
        self.history.remove_selected()
        self._sync_ui()
        self._render()

    def _undo(self) -> None:
        self.history.undo()
        self._sync_ui()
        self._render()

    def _redo(self) -> None:
        self.history.redo()
        self._sync_ui()
        self._render()

    def _clear_scene(self) -> None:
        self.history.clear()
        self._sync_ui()
        self._render()

    def _benchmark(self) -> None:
        width = max(240, min(800, self.canvas.winfo_width() or 640))
        height = max(180, min(600, self.canvas.winfo_height() or 480))
        object_count = max(1, len(self.history.state.objects))
        results = benchmark_backends(
            width=width,
            height=height,
            reps=8,
            warmup=2,
            objects=object_count,
            complexity="complex",
        )
        self.status_var.set("Benchmark complete")
        self._show_render_results(results)

    def _benchmark_schur(self) -> None:
        try:
            sizes = parse_sizes(self.schur_sizes_var.get())
            results = benchmark_schur_inverse(sizes=sizes, reps=2, warmup=1, min_block=32)
            self.status_var.set("Schur benchmark complete")
            self._show_schur_results(results)
        except Exception as error:
            self.status_var.set(str(error))
            self._set_results_table([("error", "Error", 760)], [(str(error),)])

    def _show_render_results(self, results) -> None:
        columns = [
            ("backend", "Backend", 90),
            ("mean", "Mean ms", 90),
            ("min", "Min ms", 90),
            ("max", "Max ms", 90),
            ("reps", "Reps", 70),
            ("engine", "Matrix Engine", 340),
        ]
        rows = []
        for result in results:
            if isinstance(result, RenderTiming):
                rows.append(
                    (
                        result.backend,
                        f"{result.mean_ms:.2f}",
                        f"{result.min_ms:.2f}",
                        f"{result.max_ms:.2f}",
                        str(result.reps),
                        result.matrix_engine,
                    )
                )
            else:
                backend, error = result
                rows.append((backend, "error", "-", "-", "-", error))
        self._set_results_table(columns, rows)

    def _show_schur_results(self, results) -> None:
        columns = [
            ("backend", "Backend", 80),
            ("method", "Method", 80),
            ("size", "Size", 70),
            ("mean", "Mean ms", 90),
            ("min", "Min ms", 90),
            ("max", "Max ms", 90),
            ("speedup", "Speedup", 90),
            ("residual", "Residual", 110),
            ("engine", "Matrix Engine", 300),
        ]
        baselines = {
            (result.backend, result.size): result.mean_ms
            for result in results
            if isinstance(result, SchurTiming) and result.method == "lu"
        }
        rows = []
        for result in results:
            if isinstance(result, SchurTiming):
                speedup = "-"
                baseline = baselines.get((result.backend, result.size))
                if result.method == "schur" and baseline is not None and result.mean_ms > 0.0:
                    speedup = f"{baseline / result.mean_ms:.2f}x"
                rows.append(
                    (
                        result.backend,
                        result.method,
                        str(result.size),
                        f"{result.mean_ms:.2f}",
                        f"{result.min_ms:.2f}",
                        f"{result.max_ms:.2f}",
                        speedup,
                        f"{result.residual:.2e}",
                        result.matrix_engine,
                    )
                )
            else:
                backend, error, size = result
                rows.append((backend, "error", str(size), "error", "-", "-", "-", "-", error))
        self._set_results_table(columns, rows)

    def _set_results_table(self, columns, rows) -> None:
        column_ids = [column_id for column_id, _label, _width in columns]
        for item_id in self.results_table.get_children():
            self.results_table.delete(item_id)
        self.results_table.configure(columns=column_ids)
        for column_id, label, width in columns:
            self.results_table.heading(column_id, text=label)
            self.results_table.column(column_id, width=width, minwidth=60, stretch=False, anchor=tk.W)
        for row in rows:
            self.results_table.insert("", tk.END, values=row)

    def _rotate(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> None:
        self.history.rotate_selected(dx=dx, dy=dy, dz=dz)
        self._sync_ui()
        self._render()

    def _move(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> None:
        self.history.move_selected(dx=dx, dy=dy, dz=dz)
        self._sync_ui()
        self._render()

    def _select_from_list(self, _event: tk.Event) -> None:
        selection = self.object_list.curselection()
        if not selection:
            self.history.select(None)
            self._sync_ui()
            self._render()
            return
        obj = self.history.state.objects[selection[0]]
        self.history.select(obj.id)
        self._sync_ui()
        self._render()

    def _canvas_press(self, event: tk.Event) -> None:
        object_id = self._object_id_at(event.x, event.y)
        self.history.select(object_id if object_id >= 0 else None)
        self._drag_start = (event.x, event.y)
        self._drag_base_state = self.history.state
        self._drag_has_motion = False
        self._sync_ui()
        self._render()

    def _canvas_drag(self, event: tk.Event) -> None:
        if self._drag_start is None or self._drag_base_state is None:
            return
        if self._drag_base_state.selected_id is None:
            return
        dx_pixels = event.x - self._drag_start[0]
        dy_pixels = event.y - self._drag_start[1]
        if abs(dx_pixels) + abs(dy_pixels) < 2:
            return
        scale = 4.2 / max(1, min(self.canvas.winfo_width(), self.canvas.winfo_height()))
        self.history.set_state(self._drag_base_state, record=False)
        self.history.move_selected(dx=dx_pixels * scale, dy=-dy_pixels * scale, record=False)
        self._drag_has_motion = True
        self._sync_ui()
        self._render()

    def _canvas_release(self, _event: tk.Event) -> None:
        if self._drag_has_motion and self._drag_base_state is not None:
            final_state = self.history.state
            self.history.set_state(self._drag_base_state, record=False)
            self.history.set_state(final_state, record=True)
            self._sync_ui()
        self._drag_start = None
        self._drag_base_state = None
        self._drag_has_motion = False

    def _object_id_at(self, x: int, y: int) -> int:
        if self._object_ids is None:
            return -1
        if y < 0 or x < 0 or y >= self._object_ids.shape[0] or x >= self._object_ids.shape[1]:
            return -1
        return int(self._object_ids[y, x])

    def _draw_selection_overlay(self) -> None:
        selected_id = self.history.state.selected_id
        if selected_id is None or self._object_ids is None:
            return
        mask = self._object_ids == selected_id
        if not np.any(mask):
            return
        ys, xs = np.where(mask)
        self.canvas.create_rectangle(
            int(xs.min()) - 4,
            int(ys.min()) - 4,
            int(xs.max()) + 4,
            int(ys.max()) + 4,
            outline="#f8d84a",
            width=2,
        )

    def _sync_ui(self) -> None:
        selected_id = self.history.state.selected_id
        self.object_list.delete(0, tk.END)
        selected_index = None
        for index, obj in enumerate(self.history.state.objects):
            self.object_list.insert(tk.END, f"{obj.name}  ({obj.kind})")
            if obj.id == selected_id:
                selected_index = index
        if selected_index is not None:
            self.object_list.selection_set(selected_index)
            self.object_list.see(selected_index)
        self.delete_button.configure(state=tk.NORMAL if selected_id is not None else tk.DISABLED)
        self.undo_button.configure(state=tk.NORMAL if self.history.can_undo else tk.DISABLED)
        self.redo_button.configure(state=tk.NORMAL if self.history.can_redo else tk.DISABLED)

    def _render_later(self) -> None:
        self.after_idle(self._render)

    def _render(self) -> None:
        width = max(240, self.canvas.winfo_width() or 720)
        height = max(180, self.canvas.winfo_height() or 540)
        try:
            self.renderer = Renderer(width=width, height=height, backend=self.backend_var.get())
            camera = Camera.look_at_perspective(
                eye=(2.9, 2.1, 4.4),
                target=(0.0, 0.0, 0.0),
                aspect=width / height,
                fovy_degrees=48.0,
            )
            result = self.renderer.render_many(
                self.history.render_items(include_ids=True, include_ground=True),
                camera=camera,
                backface_culling=False,
            )
            self._object_ids = result.object_ids
            self._photo = _photo_from_image(result.image)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)
            self._draw_selection_overlay()
            info = result.backend
            selected = self.history.state.selected()
            selected_text = ""
            if selected is not None:
                selected_text = f" | {selected.name} ({selected.x:.1f}, {selected.y:.1f}, {selected.z:.1f})"
            self.status_var.set(f"{info.name}: {info.matrix_engine}{selected_text}")
        except Exception as error:
            self.status_var.set(str(error))
            messagebox.showerror("Render failed", str(error))


def _photo_from_image(image) -> tk.PhotoImage:
    pixels = to_uint8(image)
    height, width, _channels = pixels.shape
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    data = header + pixels.tobytes()
    return tk.PhotoImage(data=data, format="PPM")


def main() -> None:
    app = LinGraphicsApp()
    app.mainloop()


if __name__ == "__main__":
    main()
