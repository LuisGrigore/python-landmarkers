# Landmarkers – Landmark Detection Wrapper

**Landmarkers** is a Python library that provides a simple interface for landmark detection models.  
Currently, it supports **hand landmarks using MediaPipe**, but it could support other landmarks in the future.

---

## ⚡ Features

- Hand detection in **single images** or **video frame-by-frame** (synchronous mode).  
- **Asynchronous streaming** mode with user-provided callbacks (LIVE_STREAM).  
- Results are wrapped in `HandLandmarkerResult` objects with helper methods.  
- If you want to work **directly with the raw data** returned by the model, without using the `HandLandmarkerResult` helpers, you can use the **raw result protocol** (`HandLandmarkerResultProtocol`).  
  This avoids unnecessary overhead if you don’t need drawing or helper methods.

---

## 📦 Installation

### From GitHub
You can install directly from GitHub using pip:

```bash
pip install git+aaaaaaaaaaaaaaaaaaaa.git
```

### Requirements
- Python ≥ 3.10  
- mediapipe  
- numpy
- opencv-python

---

## 🖼 Usage – Synchronous Mode

```python
import cv2
from landmarkers.hands.sync_mediapipe_hand_landmarker import SyncMediapipeHandLandmarker, MediapipeHandLandmarkerRunningMode

# Initialize detector
detector = SyncMediapipeHandLandmarker(
    model_path="hand_landmarker.task",
    running_mode=MediapipeHandLandmarkerRunningMode.IMAGE
)

# Read an image
frame = cv2.imread("hand.jpg")[:, :, ::-1]  # BGR -> RGB

# Detect hands
result = detector.detect(frame)

# Draw landmarks on the image
image_with_landmarks = result.draw(frame)
```

### Video Example
```python
cap = cv2.VideoCapture("video.mp4")
detector = SyncMediapipeHandLandmarker(
    model_path="hand_landmarker.task",
    running_mode=MediapipeHandLandmarkerRunningMode.VIDEO
)

timestamp = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = frame[:, :, ::-1]
    result = detector.detect(frame_rgb, timestamp_ms=timestamp)
    timestamp += 33  # approx. 30 FPS
```

---

## 🎥 Usage – Asynchronous Stream

### Raw Callback
```python
from landmarkers.hands.async_mediapipe_hand_landmarker import RawStreamMediapipeLandmarker

def raw_callback(result, image, timestamp_ms):
    # result is a HandLandmarkerResultProtocol object (raw model data)
    print("Detected hands:", len(result.hands))

detector = RawStreamMediapipeLandmarker(
    model_path="hand_landmarker.task",
    result_callback=raw_callback
)

# Send frames from camera
import cv2
cap = cv2.VideoCapture(0)
timestamp = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    detector.send(frame[:, :, ::-1], timestamp)
    timestamp += 33
```

### Wrapped Callback
```python
from landmarkers.hands.async_mediapipe_hand_landmarker import WrappedStreamMediapipeLandmarker

def wrapped_callback(result, image, timestamp_ms):
    print("Number of hands:", len(result.hands))
    # Draw landmarks on the image
    image_with_landmarks = result.draw(image)

detector = WrappedStreamMediapipeLandmarker(
    model_path="hand_landmarker.task",
    result_callback=wrapped_callback
)
```

---

## 🔧 Main Classes and Methods

| Class | Method | Description |
|-------|--------|------------|
| `SyncMediapipeHandLandmarker` | `detect(frame, timestamp_ms=None)` | Detects hands in a frame and returns a `HandLandmarkerResult`. |
|  | `detect_raw(frame, timestamp_ms=None)` | Returns the raw MediaPipe result (`HandLandmarkerResultProtocol`) without wrapping. |
|  | `close()` | Releases resources. |
| `RawStreamMediapipeLandmarker` | `send(frame, timestamp_ms)` | Sends a frame for asynchronous processing. |
|  | `_internal_callback(raw_result, mp_image, timestamp_ms)` | Internal callback; passes the raw result to the user callback. |
|  | `close()` | Releases resources. |
| `WrappedStreamMediapipeLandmarker` | `send(frame, timestamp_ms)` | Sends a frame for asynchronous processing. |
|  | `_internal_callback(raw_result, mp_image, timestamp_ms)` | Wraps the raw result in `HandLandmarkerResult` before calling user callback. |
|  | `close()` | Releases resources. |

---

## 🛠 Best Practices

- Always **close the detector** after use:

```python
detector.close()
```

- Provide **consistent timestamps** for video/streaming to maintain tracking.  
- `HandLandmarkerResult` provides easy access to **landmark coordinates** and **handedness**, and allows drawing landmarks directly on images.  
- If you don’t need helpers or drawing, **use the raw result protocol** to reduce overhead and work directly with the model data.
