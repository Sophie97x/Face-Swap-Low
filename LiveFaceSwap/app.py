import ssl
import sys
import threading
import time
import urllib.request
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageTk
try:
    import mediapipe as mp
except Exception:
    mp = None

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
ASSETS_DIR = APP_DIR / "assets"
SOURCE_PLACEHOLDER = ASSETS_DIR / "source_placeholder.png"
DEFAULT_CAMERA_INDEX = 0
MODEL_FILES = {
    "inswapper_128.onnx": [
        "https://raw.githubusercontent.com/sumdeusvitae/FaceSwap_v01/main/inswapper_128.onnx",
        "https://github.com/omarsar/inswapper/raw/main/inswapper_128.onnx",
        "https://huggingface.co/cloneofsimo/inswapper_v3/resolve/main/inswapper_128.onnx"
    ]
}

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class FaceSwapApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("LiveFaceSwap")
        self.geometry("1280x760")
        self.minsize(1024, 680)

        self.camera_index = DEFAULT_CAMERA_INDEX
        self.capture = None
        self.video_thread = None
        self.running = False
        self.swap_enabled = False
        self.last_frame_time = time.time()
        self.fps = 0.0
        self.latency_ms = 0
        self.source_image = None
        self.source_face = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        # MediaPipe face mesh for landmark-based warping (optional)
        self.mp_face_mesh = None
        self.source_landmarks = None
        self.delaunay_triangles = None
        self.prev_target_landmarks = None
        if mp is not None:
            try:
                self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                                                                     min_detection_confidence=0.5)
            except Exception:
                self.mp_face_mesh = None

        self.ort_session = None
        self.model_download_thread = None
        # runtime options
        self.swap_method = "Landmark"  # options: Model, Landmark, Fallback
        self.blend_strength = 0.9
        self.prev_landmarks = None
        self.landmark_smooth = 0.6
        self.build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(100, self.start_model_download)
        self.after(30, self.refresh_video_frame)

    def start_model_download(self):
        if self.model_download_thread and self.model_download_thread.is_alive():
            return
        self.model_download_thread = threading.Thread(target=self.ensure_models, daemon=True)
        self.model_download_thread.start()

    def ensure_models(self):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        for filename, urls in MODEL_FILES.items():
            path = MODELS_DIR / filename
            if path.exists():
                continue
            self.download_model(filename, urls)
        self.ort_session = self.create_onnx_session()

    def download_model(self, filename: str, urls):
        for url in urls:
            try:
                self.update_status(f"Downloading {filename}...")
                self.fetch_url_to_file(url, MODELS_DIR / filename)
                self.update_status(f"Downloaded {filename}")
                return
            except Exception:
                continue
        self.after(0, lambda: messagebox.showwarning(
            "Model Download Failed",
            f"Could not download {filename}. Place it in {MODELS_DIR} manually."
        ))

    def fetch_url_to_file(self, url: str, dest_path: Path):
        context = ssl.create_default_context()
        with urllib.request.urlopen(url, context=context, timeout=60) as response:
            if response.status != 200:
                raise RuntimeError(f"Failed to download {url}: {response.status}")
            data = response.read()
        dest_path.write_bytes(data)

    def update_status(self, text: str):
        if hasattr(self, 'status_label'):
            self.status_label.configure(text=text)

    def create_onnx_session(self):
        model_path = MODELS_DIR / "inswapper_128.onnx"
        if not model_path.exists():
            return None

        providers = []
        if sys.platform == "darwin":
            providers.append("CoreMLExecutionProvider")
        elif sys.platform == "win32":
            providers.extend(["DmlExecutionProvider", "CPUExecutionProvider"])
        else:
            providers.append("CPUExecutionProvider")

        try:
            return ort.InferenceSession(str(model_path), providers=providers)
        except Exception:
            try:
                return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            except Exception:
                return None

    def build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        source_frame = ctk.CTkFrame(self, fg_color="#1f2937", corner_radius=16)
        source_frame.grid(row=0, column=0, padx=(20, 10), pady=20, sticky="ns")
        source_frame.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(source_frame, text="Source Face", font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")
        self.source_preview = ctk.CTkLabel(source_frame, text="No image selected", width=260, height=260, fg_color="#111827", corner_radius=16)
        self.source_preview.grid(row=1, column=0, padx=20, pady=(0, 20))
        ctk.CTkButton(source_frame, text="Load Face Image", command=self.load_source_image).grid(row=3, column=0, padx=20, pady=(0, 20), sticky="ew")

        live_frame = ctk.CTkFrame(self, fg_color="#111827", corner_radius=16)
        live_frame.grid(row=0, column=1, padx=10, pady=20, sticky="nsew")
        live_frame.grid_rowconfigure(0, weight=1)
        live_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(live_frame, text="Live Feed", font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")
        self.video_label = ctk.CTkLabel(live_frame, text="Ready", fg_color="#000000", width=760, height=520, corner_radius=16)
        self.video_label.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")

        control_frame = ctk.CTkFrame(self, fg_color="#1f2937", corner_radius=16)
        control_frame.grid(row=0, column=2, padx=(10, 20), pady=20, sticky="ns")

        ctk.CTkLabel(control_frame, text="Controls", font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, padx=20, pady=(20, 12), sticky="w")
        self.start_button = ctk.CTkButton(control_frame, text="Start Webcam", command=self.toggle_webcam)
        self.start_button.grid(row=1, column=0, padx=20, pady=(0, 12), sticky="ew")
        self.swap_button = ctk.CTkSwitch(control_frame, text="Enable Face Swap", command=self.toggle_swap)
        self.swap_button.grid(row=2, column=0, padx=20, pady=(0, 12), sticky="ew")

        ctk.CTkLabel(control_frame, text="Method:", anchor="w").grid(row=5, column=0, padx=20, pady=(6,0), sticky="w")
        self.method_menu = ctk.CTkOptionMenu(control_frame, values=["Model","Landmark","Fallback"], command=self.change_method)
        self.method_menu.set(self.swap_method)
        self.method_menu.grid(row=6, column=0, padx=20, pady=(0, 12), sticky="ew")

        ctk.CTkLabel(control_frame, text="Blend Strength:", anchor="w").grid(row=7, column=0, padx=20, pady=(6,0), sticky="w")
        self.blend_slider = ctk.CTkSlider(control_frame, from_=0.0, to=1.0, number_of_steps=100, command=self.set_blend_strength)
        self.blend_slider.set(self.blend_strength)
        self.blend_slider.grid(row=8, column=0, padx=20, pady=(0, 12), sticky="ew")

        camera_options = [f"Camera {i}" for i in range(4)]
        camera_options = [f"Camera {i}" for i in range(4)]
        self.device_menu = ctk.CTkOptionMenu(control_frame, values=camera_options, command=self.change_camera)
        self.device_menu.set("Camera 0")
        self.device_menu.grid(row=3, column=0, padx=20, pady=(0, 12), sticky="ew")

        self.status_label = ctk.CTkLabel(control_frame, text="FPS: 0.0 | Latency: 0 ms", anchor="w")
        self.status_label.grid(row=4, column=0, padx=20, pady=(12, 20), sticky="ew")

        self.load_placeholder_image()

    def load_placeholder_image(self):
        if not SOURCE_PLACEHOLDER.exists():
            img = Image.new("RGB", (260, 260), color=(30, 41, 59))
            self.display_source_thumbnail(img)
            # set a fallback source face from the generated placeholder
            try:
                self.source_image = img
                self.source_face = self.extract_face_from_image(np.array(img))
            except Exception:
                self.source_face = None
            return
        img = Image.open(SOURCE_PLACEHOLDER).convert("RGB")
        self.display_source_thumbnail(img)
        try:
            self.source_image = img
            self.source_face = self.extract_face_from_image(np.array(img))
        except Exception:
            self.source_face = None

    def load_source_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*")],
            title="Select a source face image"
        )
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
            full_image = np.array(img)
            self.source_face = self.extract_face_from_image(full_image)
            if self.source_face is not None and self.source_face.size > 0:
                self.source_face = self.align_source_face(self.source_face)
                self.source_image = Image.fromarray(self.source_face)
            else:
                self.source_image = img
            # extract landmarks for landmark-based warp on the cropped source face
            try:
                source_face_rgb = self.source_face if self.source_face.ndim == 3 else full_image
                lm = self.extract_landmarks_from_image(source_face_rgb, smooth=False)
                self.prev_target_landmarks = None
                if lm is not None:
                    self.source_landmarks = lm
                    h, w = source_face_rgb.shape[:2]
                    rect = (0, 0, w, h)
                    self.delaunay_triangles = self.compute_delaunay_triangles(self.source_landmarks, rect)
                else:
                    self.source_landmarks = None
                    self.delaunay_triangles = None
            except Exception:
                self.source_landmarks = None
                self.delaunay_triangles = None
            self.display_source_thumbnail(img)
        except Exception:
            messagebox.showerror("Load Failed", "Could not open the selected image.")

    def display_source_thumbnail(self, img: Image.Image):
        thumbnail = img.copy()
        thumbnail.thumbnail((260, 260), Image.LANCZOS)
        self.source_thumb = ctk.CTkImage(light_image=thumbnail, size=(thumbnail.width, thumbnail.height))
        self.source_preview.configure(image=self.source_thumb, text="")
        self.source_preview.image = self.source_thumb

    def toggle_webcam(self):
        if self.running:
            self.running = False
            self.start_button.configure(text="Start Webcam")
            return
        self.running = True
        self.start_button.configure(text="Stop Webcam")
        self.start_video_thread()

    def toggle_swap(self, value=None):
        if value is None and hasattr(self, 'swap_button'):
            value = self.swap_button.get()
        self.swap_enabled = bool(value)

    def change_camera(self, option):
        try:
            index = int(option.split()[-1])
            self.camera_index = index
            if self.running:
                self.restart_camera()
        except (ValueError, IndexError):
            pass

    def change_method(self, value):
        try:
            self.swap_method = str(value)
        except Exception:
            pass

    def set_blend_strength(self, value):
        try:
            self.blend_strength = float(value)
        except Exception:
            pass

    def start_video_thread(self):
        if self.video_thread and self.video_thread.is_alive():
            return
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()

    def restart_camera(self):
        self.running = False
        if self.capture:
            self.capture.release()
        self.capture = None
        time.sleep(0.1)
        self.running = True
        self.start_video_thread()

    def video_loop(self):
        self.capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        if not self.capture.isOpened():
            self.running = False
            self.after(0, lambda: messagebox.showerror("Camera Error", "Unable to open the selected webcam."))
            self.after(0, lambda: self.start_button.configure(text="Start Webcam"))
            return

        while self.running:
            start_time = time.time()
            ret, frame = self.capture.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.swap_enabled and self.source_face is not None:
                frame = self.process_frame(frame)
            with self.frame_lock:
                self.latest_frame = frame
            self.update_fps_latency(start_time)

        if self.capture:
            self.capture.release()
            self.capture = None

    def refresh_video_frame(self):
        with self.frame_lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
        if frame is not None:
            image = Image.fromarray(frame)
            image = image.resize((760, 520), Image.LANCZOS)
            photo = ctk.CTkImage(light_image=image, size=(760, 520))
            self.video_label.configure(image=photo, text="")
            self.video_label.image = photo
        self.status_label.configure(text=f"FPS: {self.fps:.1f} | Latency: {self.latency_ms} ms")
        self.after(30, self.refresh_video_frame)

    def update_fps_latency(self, start_time: float):
        now = time.time()
        self.latency_ms = int((now - start_time) * 1000)
        self.fps = 1.0 / max(now - self.last_frame_time, 1e-6)
        self.last_frame_time = now

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        try:
            faces = self.detect_face(frame)
            if faces is None or len(faces) == 0:
                return frame

            x, y, w, h = faces[0]
            method = self.swap_method

            if method == "Model" and self.ort_session is not None:
                swapped = self.run_onnx_swap(frame, self.source_face, (x, y, w, h))
                if swapped is not None:
                    return swapped
                method = "Landmark"

            if method == "Landmark":
                try:
                    lm_swapped = self.landmark_swap(frame)
                    if lm_swapped is not None:
                        return lm_swapped
                except Exception:
                    pass
                method = "Fallback"

            if method == "Fallback":
                return self.paste_fallback(frame, x, y, w, h)

            return frame
        except Exception:
            return frame

    def detect_face(self, frame: np.ndarray):
        # returns list of (x,y,w,h) faces or None
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
            if len(faces) == 0:
                return None
            return faces
        except Exception:
            return None

    def extract_face_from_image(self, image: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            messagebox.showwarning("Face Not Found", "No face detected in the selected image. Using the whole image as a fallback.")
            return image
        x, y, w, h = faces[0]
        return image[y : y + h, x : x + w]

    def extract_landmarks_from_image(self, image: np.ndarray, smooth: bool = False):
        # expects RGB image ndarray
        if self.mp_face_mesh is None:
            return None
        try:
            img_rgb = image if image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.mp_face_mesh.process(img_rgb)
            if not results or not results.multi_face_landmarks:
                return None
            lm = results.multi_face_landmarks[0]
            h, w = image.shape[:2]
            pts = [(int(p.x * w), int(p.y * h)) for p in lm.landmark]
            if smooth and self.prev_target_landmarks is not None and len(self.prev_target_landmarks) == len(pts):
                smoothed = []
                alpha = self.landmark_smooth
                for (px, py), (cx, cy) in zip(self.prev_target_landmarks, pts):
                    sx = int(px * (1 - alpha) + cx * alpha)
                    sy = int(py * (1 - alpha) + cy * alpha)
                    smoothed.append((sx, sy))
                pts = smoothed
            if smooth:
                self.prev_target_landmarks = pts
            return pts
        except Exception:
            return None

    def align_source_face(self, image: np.ndarray):
        if self.mp_face_mesh is None:
            return image
        landmarks = self.extract_landmarks_from_image(image, smooth=False)
        if landmarks is None or len(landmarks) < 264:
            return image
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        rot = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        rotated = cv2.warpAffine(image, rot, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        rotated_landmarks = []
        for x, y in landmarks:
            rx = int(rot[0, 0] * x + rot[0, 1] * y + rot[0, 2])
            ry = int(rot[1, 0] * x + rot[1, 1] * y + rot[1, 2])
            rotated_landmarks.append((rx, ry))
        self.source_landmarks = rotated_landmarks
        return rotated

    def compute_delaunay_triangles(self, points, rect):
        subdiv = cv2.Subdiv2D(rect)
        for p in points:
            subdiv.insert((int(p[0]), int(p[1])))
        triangleList = subdiv.getTriangleList()
        indices = []
        for t in triangleList:
            pts = [(int(t[0]), int(t[1])), (int(t[2]), int(t[3])), (int(t[4]), int(t[5]))]
            idx = []
            for p in pts:
                for i, q in enumerate(points):
                    if abs(p[0] - q[0]) <= 1 and abs(p[1] - q[1]) <= 1:
                        idx.append(i)
                        break
            if len(idx) == 3:
                indices.append(tuple(idx))
        return indices

    def warp_triangle(self, src, dst, t_src, t_dst):
        # src/dst are images, t_src/t_dst are lists of three (x,y)
        r1 = cv2.boundingRect(np.float32([t_src]))
        r2 = cv2.boundingRect(np.float32([t_dst]))

        if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
            return

        t1Rect = []
        t2Rect = []
        for i in range(3):
            t1Rect.append(((t_src[i][0] - r1[0]), (t_src[i][1] - r1[1])))
            t2Rect.append(((t_dst[i][0] - r2[0]), (t_dst[i][1] - r2[1])))

        srcRect = src[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
        if srcRect.size == 0:
            return

        size = (r2[2], r2[3])
        mat = cv2.getAffineTransform(np.float32(t1Rect), np.float32(t2Rect))
        warped = cv2.warpAffine(srcRect, mat, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        mask = np.zeros((r2[3], r2[2], 3), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(t2Rect), (1, 1, 1), 16, 0)

        dst_rect = dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
        if dst_rect.shape == warped.shape:
            dst_rect = dst_rect * (1 - mask) + warped * mask
            dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = dst_rect

    def landmark_swap(self, frame: np.ndarray):
        if self.source_face is None or self.source_landmarks is None or self.delaunay_triangles is None or self.mp_face_mesh is None:
            return None
        try:
            target_landmarks = self.extract_landmarks_from_image(frame, smooth=True)
            if target_landmarks is None or len(target_landmarks) != len(self.source_landmarks):
                return None

            src_img = self.source_face.copy()
            if src_img.ndim == 2:
                src_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
            elif src_img.shape[2] == 4:
                src_img = cv2.cvtColor(src_img, cv2.COLOR_RGBA2BGR)
            else:
                src_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)

            dst_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out = dst_img.copy()
            src_mask = self.get_face_mask(self.source_landmarks, src_img.shape[:2])
            target_mask = self.get_face_mask(target_landmarks, dst_img.shape[:2])
            if src_mask is not None and target_mask is not None:
                src_img = self.color_correct(src_img, dst_img, src_mask, target_mask)

            aligned_src_img, aligned_source_landmarks = self.align_source_to_target(src_img, self.source_landmarks, target_landmarks, dst_img.shape[:2])
            if aligned_src_img is not None and aligned_source_landmarks is not None:
                src_img = aligned_src_img
                source_landmarks = aligned_source_landmarks
            else:
                source_landmarks = self.source_landmarks

            for tri in self.delaunay_triangles:
                t_src = [source_landmarks[i] for i in tri]
                t_dst = [target_landmarks[i] for i in tri]
                if not self.is_valid_triangle(t_src) or not self.is_valid_triangle(t_dst):
                    continue
                self.warp_triangle(src_img, out, t_src, t_dst)

            hull = self.get_face_hull(target_landmarks)
            if hull is None or len(hull) == 0:
                return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

            mask = np.zeros(dst_img.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull, 255)
            mask = cv2.GaussianBlur(mask, (31, 31), 0)
            center = tuple(np.mean(hull.reshape(-1, 2), axis=0).astype(int))
            try:
                cloned = cv2.seamlessClone(out, dst_img, mask, center, cv2.NORMAL_CLONE)
                res = cv2.cvtColor(cloned, cv2.COLOR_BGR2RGB)
            except Exception:
                res = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

            if 0.0 <= self.blend_strength < 1.0:
                res = cv2.addWeighted(frame, 1.0 - self.blend_strength, res, self.blend_strength, 0)
            return res
        except Exception:
            return None

    def paste_fallback(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        try:
            src = cv2.resize(self.source_face, (w, h))
            if src.dtype != np.uint8:
                src = np.clip(src * 255.0, 0, 255).astype(np.uint8)
            if src.ndim == 2:
                src = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
            src_bgr = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            mask = 255 * np.ones(src_bgr.shape[:2], dtype=np.uint8)
            center = (x + w // 2, y + h // 2)
            out = frame.copy()
            try:
                cloned = cv2.seamlessClone(src_bgr, frame_bgr, mask, center, cv2.NORMAL_CLONE)
                out = cv2.cvtColor(cloned, cv2.COLOR_BGR2RGB)
            except Exception:
                alpha = self.blend_strength
                out[y:y+h, x:x+w] = cv2.addWeighted(out[y:y+h, x:x+w], 1-alpha, src, alpha, 0)
            return out
        except Exception:
            return frame

    def run_onnx_swap(self, frame: np.ndarray, source_face: np.ndarray, bbox=None) -> np.ndarray:
        source_resized = cv2.resize(source_face, (128, 128)).astype(np.float32) / 255.0
        target_resized = cv2.resize(frame, (128, 128)).astype(np.float32) / 255.0

        inputs = self.ort_session.get_inputs()
        feed = {}

        if len(inputs) == 1:
            combined = np.concatenate([target_resized, source_resized], axis=2)
            if inputs[0].shape[1] in (3, 6):
                feed[inputs[0].name] = np.transpose(combined, (2, 0, 1))[None, ...]
            else:
                feed[inputs[0].name] = combined[None, ...]
        elif len(inputs) == 2:
            feed[inputs[0].name] = np.transpose(target_resized, (2, 0, 1))[None, ...]
            feed[inputs[1].name] = np.transpose(source_resized, (2, 0, 1))[None, ...]
        else:
            feed[inputs[0].name] = np.transpose(target_resized, (2, 0, 1))[None, ...]

        try:
            output = self.ort_session.run(None, feed)[0]
            output_image = self.decode_output(output, frame.shape)
            if output_image is None:
                return None

            # If bbox provided, paste the resized output into the bbox area
            if bbox is not None:
                x, y, w, h = bbox
                face = cv2.resize(output_image, (w, h))
                out = frame.copy()
                # simple seamless paste via weighted blend
                alpha = 0.95
                try:
                    out[y:y+h, x:x+w] = cv2.addWeighted(out[y:y+h, x:x+w], 1-alpha, face, alpha, 0)
                except Exception:
                    out[y:y+h, x:x+w] = face
                return out

            return output_image
        except Exception:
            return None

    def decode_output(self, output: np.ndarray, target_shape):
        if output is None:
            return None
        output = np.asarray(output)
        if output.ndim == 4:
            output = output[0]
        if output.dtype != np.uint8:
            output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        if output.shape[0] in (1, 3):
            output = np.transpose(output, (1, 2, 0))
        output = cv2.resize(output, (target_shape[1], target_shape[0]))
        return output

    def is_valid_triangle(self, triangle):
        try:
            return abs(cv2.contourArea(np.array(triangle, dtype=np.float32))) > 1.0
        except Exception:
            return False

    def get_face_hull(self, landmarks):
        try:
            hull = cv2.convexHull(np.array(landmarks, dtype=np.int32))
            return hull.reshape(-1, 2)
        except Exception:
            return None

    def get_face_mask(self, landmarks, shape):
        try:
            hull = self.get_face_hull(landmarks)
            if hull is None or len(hull) == 0:
                return None
            mask = np.zeros(shape, dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull, 255)
            blurred = cv2.GaussianBlur(mask, (31, 31), 0)
            return blurred
        except Exception:
            return None

    def get_alignment_points(self, landmarks):
        try:
            def avg(indices):
                pts = [landmarks[i] for i in indices]
                x = sum([p[0] for p in pts]) / len(pts)
                y = sum([p[1] for p in pts]) / len(pts)
                return (x, y)

            left_eye = avg([33, 133])
            right_eye = avg([362, 263])
            nose = landmarks[1]
            mouth_left = landmarks[61]
            mouth_right = landmarks[291]
            return [left_eye, right_eye, nose, mouth_left, mouth_right]
        except Exception:
            return None

    def align_source_to_target(self, src_img, source_landmarks, target_landmarks, target_shape):
        try:
            src_points = self.get_alignment_points(source_landmarks)
            dst_points = self.get_alignment_points(target_landmarks)
            if src_points is None or dst_points is None:
                return None, None

            src_arr = np.array(src_points, dtype=np.float32)
            dst_arr = np.array(dst_points, dtype=np.float32)
            M, _ = cv2.estimateAffinePartial2D(src_arr, dst_arr, method=cv2.LMEDS)
            if M is None:
                return None, None

            warped = cv2.warpAffine(src_img, M, (target_shape[1], target_shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            transformed_landmarks = [tuple(np.dot(np.append(np.array(p, dtype=np.float32), 1.0), M).tolist()) for p in source_landmarks]
            return warped, transformed_landmarks
        except Exception:
            return None, None

    def color_correct(self, src, dst, src_mask, dst_mask):
        try:
            src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
            dst_lab = cv2.cvtColor(dst, cv2.COLOR_BGR2LAB).astype(np.float32)
            src_mean, src_std = cv2.meanStdDev(src_lab, mask=src_mask)
            dst_mean, dst_std = cv2.meanStdDev(dst_lab, mask=dst_mask)
            corrected = src_lab.copy()
            for i in range(3):
                if src_std[i][0] > 1e-6:
                    corrected[:, :, i] = (corrected[:, :, i] - src_mean[i][0]) * (dst_std[i][0] / src_std[i][0]) + dst_mean[i][0]
            corrected = np.clip(corrected, 0, 255).astype(np.uint8)
            return cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)
        except Exception:
            return src

    def on_close(self):
        self.running = False
        if self.capture:
            self.capture.release()
        self.destroy()


if __name__ == "__main__":
    app = FaceSwapApp()
    app.mainloop()
