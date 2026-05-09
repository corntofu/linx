"""Tkinter GUI for creating and editing lingraphics scenes."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

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
        self.status_var = tk.StringVar(value="")
        self.history = SceneHistory()
        self.renderer: Renderer | None = None
        self._photo: tk.PhotoImage | None = None

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
            values=("cube", "pyramid"),
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

        rotate_box = ttk.LabelFrame(sidebar, text="Rotate Selected", padding=8)
        rotate_box.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(rotate_box, text="X", command=lambda: self._rotate(dx=15)).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(rotate_box, text="Y", command=lambda: self._rotate(dy=15)).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(rotate_box, text="Z", command=lambda: self._rotate(dz=15)).grid(row=0, column=2, sticky="ew", padx=(4, 0))
        rotate_box.columnconfigure(0, weight=1)
        rotate_box.columnconfigure(1, weight=1)
        rotate_box.columnconfigure(2, weight=1)

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

        self.bind("<Command-z>", lambda _event: self._undo())
        self.bind("<Control-z>", lambda _event: self._undo())
        self.bind("<BackSpace>", lambda _event: self._delete_selected())
        self.bind("<Delete>", lambda _event: self._delete_selected())

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

    def _rotate(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> None:
        self.history.rotate_selected(dx=dx, dy=dy, dz=dz)
        self._sync_ui()
        self._render()

    def _select_from_list(self, _event: tk.Event) -> None:
        selection = self.object_list.curselection()
        if not selection:
            self.history.select(None)
            self._sync_ui()
            return
        obj = self.history.state.objects[selection[0]]
        self.history.select(obj.id)
        self._sync_ui()

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
            result = self.renderer.render_many(self.history.render_items(), camera=camera)
            self._photo = _photo_from_image(result.image)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)
            info = result.backend
            self.status_var.set(f"{info.name}: {info.matrix_engine}")
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
