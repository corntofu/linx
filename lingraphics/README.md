# lingraphics

`lingraphics`는 `numpy`와 이 워크스페이스의 `linx`를 토글해서 쓸 수 있는 작은 소프트웨어 그래픽스 렌더러입니다. 삼각형 메시, 카메라 행렬, perspective projection, z-buffer, 간단한 Lambert 조명을 포함합니다.

## 실행

먼저 NumPy가 보이는 Python을 사용하세요. 이 머신에서는 아래 명령으로 검증했습니다.

```bash
/opt/anaconda3/bin/python3 -c "import numpy"
```

```bash
cd lingraphics
/opt/anaconda3/bin/python3 -m lingraphics --backend numpy --output renders/cube_numpy.ppm
/opt/anaconda3/bin/python3 -m lingraphics --backend linx --output renders/cube_linx.ppm
```

또는:

```bash
cd lingraphics
/opt/anaconda3/bin/python3 examples/render_demo.py --backend auto --mesh pyramid --output renders/pyramid.ppm
```

`--backend` 값은 `auto`, `numpy`, `linx` 중 하나입니다. `auto`는 `linx`를 먼저 시도하고 실패하면 `numpy`로 내려갑니다.

## GUI 앱

아래 명령은 창을 열고 도형을 직접 추가/삭제/회전할 수 있는 Tkinter 앱을 실행합니다.

```bash
cd lingraphics
/opt/anaconda3/bin/python3 app.py
```

동일하게 `/opt/anaconda3/bin/python3 -m lingraphics.gui_app`로도 실행할 수 있습니다.

앱에서 가능한 동작:

- `Add`: cube 또는 pyramid 생성
- `Delete`: 선택한 도형 제거
- `Undo` / `Redo`: 생성, 제거, 회전, 전체 삭제 되돌리기/다시 실행
- `X` / `Y` / `Z`: 선택한 도형 회전
- `Backend`: `auto`, `numpy`, `linx` 렌더링 백엔드 토글

키보드 단축키는 `Cmd+Z` 또는 `Ctrl+Z` 실행 취소, `Delete`/`Backspace` 삭제입니다.

## Apple M2 최적화 메모

- `linx` 백엔드는 행렬곱/변환을 기존 `linx` C++ 확장으로 넘깁니다. 현재 로컬 빌드는 `Apple Accelerate BLAS/LAPACK (arm64)`로 확인됩니다.
- `numpy` 백엔드는 설치된 NumPy의 벡터화/BLAS 경로를 사용합니다. 현재 환경은 arm64 OpenBLAS 빌드입니다.
- 중요한 점: NumPy와 현재 `linx`는 Python API 수준에서 Apple GPU를 직접 예약하지 않습니다. 즉 "M2 GPU 직접 실행"이라고 과장하지 않고, M2에서 가능한 SIMD/BLAS/Accelerate 최적화 경로를 사용하도록 구성했습니다. 진짜 Apple GPU/Metal 커널이 필요하면 이후 `mlx` 또는 Metal backend를 세 번째 백엔드로 추가하는 구조입니다.

## API 예시

```python
from lingraphics import Camera, Mesh, Renderer, rotate_x, rotate_y

renderer = Renderer(width=512, height=512, backend="linx")
camera = Camera.look_at_perspective(aspect=1.0)
model = rotate_y(0.6) @ rotate_x(0.3)

result = renderer.render(Mesh.cube(), camera=camera, model=model)
result.save("renders/cube.ppm")
```

## 테스트

```bash
cd lingraphics
/opt/anaconda3/bin/python3 -m unittest discover -s tests
```
