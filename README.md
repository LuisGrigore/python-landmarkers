
# Landmarkers – Modular Landmark Detection Library

**Landmarkers** is a Python library that provides a modular interface for landmark detection models, with built-in visualization capabilities.  
Currently, it supports **hand landmarks using MediaPipe**, but its architecture allows for easy extension to other landmark types and models.


## Features

- Modular architecture with base classes for different running modes: **Image**, **Video**, and **Live Stream**.
- **Inference objects** that encapsulate landmark data, world landmarks, and metadata.
- **Sequences** for handling temporal data and sequences of inferences.
- **Visualization system** with composable layers (points, centroids, bounding boxes, sequences) and viewers.
- Results wrapped in structured objects with helper methods.
- Compatibility with Python's **context API** for automatic resource management.
- Extensible design for adding new landmark types and models.


## Installation

You can install directly from GitHub using pip:

```bash
pip install git+https://github.com/LuisGrigore/python-landmarkers.git
```

### Requirements

- Python ≥ 3.10

### Dependencies

#### Running

| Name | Version used |
|------|--------------|
| mediapipe | 0.10.31 |
| numpy | 2.2.6 |
| opencv-python | 4.12.0.88 |

#### Development

| Name | Version used |
|------|--------------|
| build | 1.3.0 |
| pytest | 9.0.2 |

## Usage – Synchronous Mode

### Image

```python
from landmarkers.mp.hands import MPImageLandmarker
import cv2

image = cv2.imread("image.jpg")

with MPImageLandmarker(model_path='hand_landmarker.task') as landmarker:
    inferences = landmarker.infer(image)
    if inferences:
        for inference in inferences:
            print(f"Detected hand with {len(inference.landmarks)} landmarks")
            # Access landmarks: inference.landmarks
            # Access world landmarks: inference.world_landmarks
            # Access metadata: inference.metadata (handedness info)
```

### Video

```python
from landmarkers.mp.hands import MPVideoLandmarker
import cv2
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

with MPVideoLandmarker(model_path='hand_landmarker.task') as landmarker:
	get_timestamp_ms = timestamper_ms(time.time())
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break

		inferences = landmarker.infer(frame, get_timestamp_ms())
		if inferences:
			for inference in inferences:
				print(f"Detected hand at {get_timestamp_ms()}ms")

		cv2.imshow('Video', frame)
		if cv2.waitKey(10) & 0xFF == ord('q'):
			break

cap.release()
cv2.destroyAllWindows()
```

## Usage – Asynchronous Mode

### Live Stream

```python
from landmarkers.mp.hands import MPLiveStreamLandmarker
import cv2
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

def my_callback(inferences, image, timestamp_ms):
	print(f"Detected {len(inferences)} hands at {timestamp_ms}ms")
	# Process inferences here

cap = cv2.VideoCapture(0)

with MPLiveStreamLandmarker(model_path='hand_landmarker.task', callback=my_callback) as detector:
	get_timestamp_ms = timestamper_ms(time.time())
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break
		detector.infer(frame, get_timestamp_ms())
		if cv2.waitKey(10) & 0xFF == ord('q'):
			break

cap.release()
cv2.destroyAllWindows()
```

## Visualization

The library includes a flexible visualization system for rendering landmark data on images.

```python
from landmarkers.visualization import Viewer
from landmarkers.visualization.layers import PointsLayer, CentroidLayer, BBoxLayer
from landmarkers.visualization.visualizers import LandmarksVisualizer
from landmarkers.landmarks import Landmarks
import cv2

# Assuming you have inferences from a landmarker
inferences = landmarker.infer(image)

if inferences:
	# Create a viewer with multiple layers
	viewer = (Viewer.builder()
	          .add_layer(PointsLayer(color=(0, 255, 0), radius=3))
	          .add_layer(CentroidLayer(color=(255, 0, 0), radius=5))
	          .add_layer(BBoxLayer(color=(0, 0, 255), thickness=2))
	          .build())

	# Create landmarks from inference
	landmarks = Landmarks(inferences[0].landmarks)
	visualizer = LandmarksVisualizer(landmarks)

	# Render on image
	rendered_image = viewer.render(visualizer, background=image)
	cv2.imshow('Landmarks', rendered_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
```
```python
	# Render on image
	rendered_image = viewer.render(visualizer, background=image)
	cv2.imshow('Landmarks', rendered_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
```

## API

### Main Classes and Methods

#### Base Landmarkers

| Class | Method | Description |
|-------|--------|------------|
| `BaseLandmarker` | `close()` | Releases resources. |
| `ImageLandmarker` | `infer(image)` | Detects landmarks in a single image. |
| `VideoLandmarker` | `infer(image, timestamp_ms)` | Detects landmarks in a video frame. |
| `LiveStreamLandmarker` | `infer(image, timestamp_ms)` | Sends frame for asynchronous processing. |

#### MediaPipe Hands Implementation

| Class | Description |
|-------|-------------|
| `MPImageLandmarker` | MediaPipe hand landmarker for images. |
| `MPVideoLandmarker` | MediaPipe hand landmarker for video. |
| `MPLiveStreamLandmarker` | MediaPipe hand landmarker for live streaming. |

#### Inference and Data Structures

| Class | Description |
|-------|-------------|
| `Inference[M]` | Contains landmarks, world landmarks, and metadata. |
| `Landmarks` | Represents a set of 2D/3D landmark points. |
| `LandmarksSequence` | Sequence of landmarks over time. |
| `InferenceSequence[M]` | Sequence of inferences. |

#### Visualization

| Class | Description |
|-------|-------------|
| `Viewer[T]` | Renders data using layers on an image. |
| `ViewerBuilder[T]` | Builder for creating viewers with layers. |
| `Layer[T]` | Base class for visualization layers. |
| `PointsLayer` | Draws landmark points. |
| `CentroidLayer` | Draws centroid of landmarks. |
| `BBoxLayer` | Draws bounding box. |
| `SequenceLayer` | Draws sequences with time-based effects. |

#### Protocols

| Protocol | Methods |
|----------|---------|
| `HasPoints` | `points()` - Returns list of points. |
| `HasCentroid` | `centroid()` - Returns centroid. |
| `HasBBox` | `bbox()` - Returns bounding box. |
| `HasSequence` | `sequence()` - Returns sequence data. |


## Best Practices

- Always **close the landmarker** after use or use it as a context manager:

```python
with MPImageLandmarker(model_path='model.task') as landmarker:
	inferences = landmarker.infer(image)
# Resources are automatically released
```

- Provide **consistent timestamps** for video/streaming to maintain tracking.
- Use the **visualization system** for consistent rendering of landmark data.
- Leverage **sequences** for temporal analysis of landmark data.
- The modular design allows for **easy extension** to new landmark types by implementing the base classes.