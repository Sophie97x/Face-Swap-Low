# Face-Swap-Low

A portable real-time face swap application built with Python, OpenCV, MediaPipe, and CustomTkinter.

## Features

- Real-time webcam preview with face swap toggle
- Landmark-based face warping and blending
- Fallback seamless clone mode
- Camera selection, FPS/latency display, and blend strength control

## Requirements

- Python 3.10+ recommended
- `customtkinter`
- `opencv-python`
- `onnxruntime`
- `Pillow`
- `numpy`
- `mediapipe`

## Setup

1. Open a terminal in `LiveFaceSwap/`
2. Create and activate the virtual environment:
   - Windows:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## Run

From `LiveFaceSwap/`:

```powershell
.\.venv\Scripts\python.exe app.py
```

## Usage

1. Click `Start Webcam`
2. Click `Load Face Image` and select a frontal face picture
3. Enable the `Face Swap` switch
4. Choose `Method: Landmark` for the current pipeline
5. Adjust `Blend Strength` as needed

## Notes

- The current implementation is primarily landmark-based and may require a clean frontal source image for the best result.
- If `DmlExecutionProvider` is unavailable, ONNXRuntime will fall back to CPU execution.

## License

This project is provided for experimentation and learning.
