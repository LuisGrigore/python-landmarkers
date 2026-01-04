
# Landmarkers – Landmark Detection Wrapper

**Landmarkers** is a Python library that provides a simple interface for landmark detection models.  
Currently, it supports **hand landmarks using MediaPipe**, but it could support other landmarks in the future.


## Features

- Hand detection in **single images** or **video frame-by-frame** (synchronous mode).  
- **Asynchronous streaming** mode with user-provided callbacks (LIVE_STREAM).  
- Results are wrapped in `HandLandmarkerResult` objects with helper methods.  
- If you want to work **directly with the raw data** returned by the model, without using the `HandLandmarkerResult` helpers, you can use the **raw result protocol** (`HandLandmarkerResultProtocol`).  
  This avoids unnecessary overhead if you don’t need drawing or helper methods.
- Compatibility with pythons **context API**.


## Installation

You can install directly from GitHub using pip:

```bash
pip install git+https://github.com/LuisGrigore/python-landmarkers.git
```

### Requirements

- Python ≥ 3.10
## Dependencies

### Running

| Name | Version used |
|------|--------------|
| mediapipe | 0.10.31 |
|numpy | 2.2.6 |
|opencv-python | 4.12.0.88|


### Development

| Name | Version used |
|------|--------------|
| build | 1.3.0 |
| pytest | 9.0.2 |

## Usage – Synchronous Mode

### Image

```python
from landmarkers.hands import SyncMediapipeHandLandmarker, MediapipeHandLandmarkerRunningMode as RunningMode
import cv2

image = cv2.imread("image.jpg")

with SyncMediapipeHandLandmarker(model_path='hand_landmarker.task',
                                 running_mode=RunningMode.IMAGE) as hand_landmarker:
		result = hand_landmarker.detect(image)
		image_with_landmarks = result.draw(image)
        cv2.imshow("Example", imagen)

cv2.waitKey(0)
cv2.destroyAllWindows()

```

### Video

```python
from landmarkers.hands import SyncMediapipeHandLandmarker, HandLandmarkerResult, MediapipeHandLandmarkerRunningMode as RunningMode
import time

def timestamper_ms(current_time: float):
	last_timestamp = int(current_time * 1000)
	def get_timestamp_ms():
		nonlocal last_timestamp
		timestamp = int(time.time() * 1000)
		if timestamp <= last_timestamp:
			timestamp = last_timestamp + 1
		last_timestamp = timestamp
		return timestamp
	return get_timestamp_ms

cap = cv2.VideoCapture(0)


with SyncMediapipeHandLandmarker(model_path='hand_landmarker.task',
                                 running_mode=RunningMode.VIDEO) as hand_landmarker:
	get_timestamp_ms = timestamper_ms(time.time())
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:

		result = hand_landmarker.detect(frame,get_timestamp_ms())
		frame = result.draw(frame)

		cv2.imshow('Example', frame)
		if cv2.waitKey(10) & 0xFF == ord('q'):
			break

cap.release()
cv2.destroyAllWindows()


```
## API Reference

### Main Classes and Methods

| Class | Method | Description |
|-------|--------|------------|
| `SyncMediapipeHandLandmarker` | `detect(frame, timestamp_ms=None)` | Detects hands in a frame and returns a `HandLandmarkerResult`. |
|  | `detect_raw(frame, timestamp_ms=None)` | Returns the raw MediaPipe result (`HandLandmarkerResultProtocol`) without wrapping. |
|  | `close()` | Releases resources. |
| `RawStreamMediapipeLandmarker` | `send(frame, timestamp_ms)` | Sends a frame for asynchronous processing. |
|  | `close()` | Releases resources. |
| `WrappedStreamMediapipeLandmarker` | `send(frame, timestamp_ms)` | Sends a frame for asynchronous processing. |
|  | `close()` | Releases resources. |
| `HandLandmarkerResult` | `hands_count` | Number of detected hands. |
|  | `time_stamp_ms` | Timestamp of the result in milliseconds. |
|  | `draw(image, hand_index=None)` | Draws hand landmarks on the provided image. |
|  | `landmarks_array(hand_index=None)` | Returns landmarks as a numpy array. |
|  | `world_landmarks_array(hand_index=None)` | Returns world landmarks as a numpy array. |
|  | `landmarks_array_relative_to_wrist(hand_index=None)` | Returns landmarks relative to the wrist. |
|  | `handedness(hand_index=None)` | Returns handedness information as a numpy array. |
|  | `data` | Raw MediaPipe result data. |


## Best Practices

- Always **close the detector** after use:

```python
detector.close()
```

- Provide **consistent timestamps** for video/streaming to maintain tracking.  
- `HandLandmarkerResult` provides easy access to **landmark coordinates** and **handedness**, and allows drawing landmarks directly on images.  
- If you don’t need helpers or drawing, **use the raw result protocol** to reduce overhead and work directly with the model data.