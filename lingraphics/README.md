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

- `Add`: cube, pyramid, sphere, torus, tralalero 생성
- `Delete`: 선택한 도형 제거
- `Undo` / `Redo`: 생성, 제거, 회전, 전체 삭제 되돌리기/다시 실행
- `X` / `Y` / `Z`: 선택한 도형 회전
- `Move Selected`: 선택한 도형을 좌/우/상/하/앞/뒤로 이동
- 캔버스 클릭: 해당 도형 선택
- 캔버스 드래그: 선택한 도형 이동
- `Backend`: `auto`, `numpy`, `linx` 렌더링 백엔드 토글

키보드 단축키는 `Cmd+Z` 또는 `Ctrl+Z` 실행 취소, `Delete`/`Backspace` 삭제, 방향키 카메라 뷰 회전, `W`/`S` 선택 도형 앞뒤 이동입니다.

그래픽은 단색 배경 대신 그라데이션/비네트 배경, rim light 느낌의 추가 음영, 선택 도형의 노란 외곽 표시를 사용합니다. 클릭 선택은 렌더러가 같이 만드는 픽셀별 object-id 버퍼로 처리합니다.

현재 렌더러는 조금 더 사실적인 느낌을 위해 diffuse + specular highlight + rim light + depth fog를 섞어 사용하고, GUI/CLI 데모에는 바닥면을 함께 렌더링합니다.

## Schur inverse 그래픽스

렌더러는 조명용 normal matrix를 계산할 때 `inverse(model_3x3).T`가 필요합니다. 이 역행렬 계산은 `backend.inverse(..., method="schur")`를 통해 지나갑니다.

- `linx` 백엔드: 기존 `linx.inverse(method="schur")` 또는 `linx.inverse_schur` 사용
- `numpy` 백엔드: 순수 NumPy로 구현한 재귀 Schur complement inverse 사용

이렇게 해서 비균일 스케일/회전이 들어간 도형도 Schur inverse 기반 normal matrix로 조명됩니다.

## 렌더링 시간 측정

같은 씬을 `numpy`와 `linx` 백엔드로 각각 여러 번 렌더링하고 평균/최소/최대 시간을 출력합니다.

```bash
cd lingraphics
/opt/anaconda3/bin/python3 -m lingraphics.benchmark --width 640 --height 480 --objects 8 --reps 20
```

복잡한 도형에 대한 벤치마크도 지원합니다. `complex` 모드는 sphere/torus처럼 삼각형 수가 많은 메시를 사용합니다.

`tralalero`는 Tralalero Tralala를 stylized mesh로 만든 도형입니다. 파란 상어 몸통, 세 개의 지느러미 다리, 밝은 운동화 실루엣을 조합했고 특정 브랜드 로고는 넣지 않았습니다.

```bash
cd lingraphics
/opt/anaconda3/bin/python3 -m lingraphics.benchmark --width 160 --height 120 --objects 2 --reps 5 --complexity complex
```

`--complexity` 값은 `basic`, `mixed`, `complex` 중 하나입니다.

GUI에서도 `Benchmark Render` 버튼으로 짧은 렌더 벤치마크를 실행할 수 있습니다. 결과는 알림창이 아니라 캔버스 아래 `Benchmark Results` 테이블에 표시됩니다.

## 3초 애니메이션 렌더링

Tralalero Tralala가 cube를 갖고 노는 3초짜리 animated GIF를 렌더링합니다. 이 환경에는 `ffmpeg`가 없어서 MP4 대신 GIF로 저장합니다.

```bash
cd lingraphics
/opt/anaconda3/bin/python3 -m lingraphics.video --backend linx --width 320 --height 240 --fps 10 --duration 3 --output renders/tralalero_cube.gif
```

명령은 전체 렌더 시간, 프레임당 평균 시간, GIF 인코딩 시간을 출력합니다.

세 모드 `numpy`, `linx`, `linx-schur`로 같은 3초 애니메이션을 렌더링하고 CSV 표로 비교하려면:

```bash
cd lingraphics
/opt/anaconda3/bin/python3 examples/time_tralalero_cube_video_csv.py --width 320 --height 240 --fps 10 --duration 3 --csv-output renders/video_compare/timing.csv
```

Schur inverse가 빨라지는 구간을 찾고 싶다면 렌더 benchmark 대신 Schur 전용 모드를 쓰세요. 렌더러의 normal matrix는 3x3이라 Schur의 장점이 잘 드러나지 않지만, 큰 정방행렬 inverse에서는 구간을 직접 조절해 비교할 수 있습니다.

```bash
cd lingraphics
/opt/anaconda3/bin/python3 -m lingraphics.benchmark --mode schur --schur-sizes 64,128,256,512 --schur-reps 3 --schur-warmup 1 --schur-min-block 32
```

`--schur-sizes`는 쉼표로 구분한 정방행렬 크기입니다. 실행 시간이 길어지면 `64,128,256`처럼 줄이고, Schur가 유리한 지점을 더 보고 싶으면 `512,1024`를 추가하세요. GUI에서는 `Schur sizes` 입력칸에 같은 형식으로 크기를 넣고 `Benchmark Schur`를 누르면 앱 내부 테이블에 표시됩니다.

NumPy inverse, linx inverse, linx Schur inverse만 직접 비교하려면 아래 예제 스크립트를 쓰면 됩니다.

```bash
cd lingraphics
/opt/anaconda3/bin/python3 examples/compare_numpy_linx_schur.py --sizes 64,128,256 --reps 3 --warmup 1 --min-block 32
```

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
