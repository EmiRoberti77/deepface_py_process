# 🧠 DeepFace Face Verification and Model Evaluation Guide

This project demonstrates how to use the **DeepFace** Python framework to perform face verification and evaluate multiple built-in face recognition models. It also helps developers select the best model based on speed, embedding size, backend, and verification accuracy.

---

## 📖 What is DeepFace?

**DeepFace** is a lightweight face recognition and analysis framework for Python. It wraps several state-of-the-art deep learning models for face recognition and verification, offering a simple interface for complex tasks like identity matching, facial attribute analysis, and more.

It supports multiple backends and provides consistent embeddings, facial verification (`verify`), and face search (`find`) APIs.

---

## 📂 Project Structure

```
deepface_sample/
├── images/
│   ├── face1.png
│   └── face2.png
├── my_db/
│   ├── face1.png
│   └── face2.png
├── main.py
└── requirements.txt
```

---

## 📝 Files Description

### `main.py`

```python
from deepface import DeepFace

print("starting face comparison")

result = DeepFace.verify(
  img1_path='./images/face1.png',
  img2_path='./images/face2.png'
)

print("completed face comparison")
print(result)
```

---

## ⚙️ Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
```

### 2. Install Requirements

```bash
pip install tensorflow==2.12.0 deepface numpy pandas opencv-python
```

_If using TensorFlow 2.19+, also install:_

```bash
pip install tf-keras
```

---

## 🚀 Run the Face Verification

Ensure two images are available in the `images/` directory:

```bash
python face_comparison.py
```

Expected output:

```
starting face comparison
completed face comparison
{ 'verified': True, 'distance': 0.23, ... }
```

---

## 🔍 DeepFace Supported Models – Explained

| Model          | Embedding Size | Backend    | Highlights                                                                |
| -------------- | -------------- | ---------- | ------------------------------------------------------------------------- |
| **VGG-Face**   | 2622           | Keras      | One of the earliest public CNN face models. Larger embedding, slower.     |
| **Facenet**    | 128            | TensorFlow | High performance on LFW; small and efficient embeddings. Good for mobile. |
| **Facenet512** | 512            | TensorFlow | Higher dimensional version of Facenet. Improved accuracy.                 |
| **OpenFace**   | 128            | Torch      | Lightweight academic model; fast inference.                               |
| **DeepID**     | 160            | Torch      | Older model, very lightweight and quick, but lower accuracy.              |
| **ArcFace**    | 512            | MXNet      | Strong accuracy, SOTA on many benchmarks. Great for identity matching.    |
| **Dlib**       | 128            | C++ (dlib) | C++-based, fast and accurate; good for traditional setups.                |
| **SFace**      | 128            | PyTorch    | Samsung model; robust performance and low latency.                        |

---

## ⚡ Speed and Accuracy Benchmark

This script evaluates models by comparing:

- **Direct cosine distance from embeddings**
- **Standard `DeepFace.verify()` function**

```python
from deepface import DeepFace
import numpy as np
import time

_IMAGE_1 = './images/face1.png'
_IMAGE_2 = './images/face2.png'
_DESCRIPTION_1 = "Manual cosine distance comparison"
_DESCRIPTION_2= "DeepFace verify image comparison"

def cosine_distance(vec1, vec2):
  try:
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return 1 - (np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
  except Exception as e:
    print(e)


def face_comparison(img1_path, img2_path, model_name):
  """
  direct vector comparison by extracting the vector value from the image
  """
  model = DeepFace.build_model(model_name=model_name)
  embedding1 = DeepFace.represent(img_path=img1_path, model_name=model_name)[0]["embedding"]
  embedding2 = DeepFace.represent(img_path=img2_path, model_name=model_name)[0]["embedding"]
  return cosine_distance(vec1=embedding1, vec2=embedding2)

def verify(img1_path, img2_path, model_name):
  """
  deepface comparison of faces by passing the whole image into the method
  """
  try:
    result = DeepFace.verify(img1_path=img1_path, img2_path=img2_path, model_name=model_name)
    return result["distance"], result["verified"]
  except Exception as e:
    print(e)

def time_function(label, description, func, *args, **kwargs):
  start = time.time()
  result = func(*args, **kwargs)
  end = time.time()
  print(f"{label}, {description} took:{end-start:.3f} seconds")
  return result



if __name__ == "__main__":
  model_names = [
        "VGG-Face", "Facenet", "Facenet512", "OpenFace",
        "DeepID", "ArcFace", "Dlib", "SFace"
    ]

  for model_name in model_names:
    print("=======================")
    distance = time_function(model_name, _DESCRIPTION_1, face_comparison, _IMAGE_1, _IMAGE_2, model_name)
    print(distance)
    result, verified = time_function(model_name, _DESCRIPTION_2, verify, _IMAGE_1, _IMAGE_2, model_name)
    print(result, verified)
    print("=======================")
```

### 🧪 Sample Benchmark Results

```bash
=======================
VGG-Face, Manual cosine distance comparison took:1.432 seconds
0.6977465574818573
VGG-Face, DeepFace verify image comparison took:0.306 seconds
0.6977465574818573 False
=======================
=======================
Facenet, Manual cosine distance comparison took:1.604 seconds
0.3555793982961579
Facenet, DeepFace verify image comparison took:0.293 seconds
0.3555793982961579 True ✅
=======================
=======================
Facenet512, Manual cosine distance comparison took:1.304 seconds
0.40204405347090477
Facenet512, DeepFace verify image comparison took:0.281 seconds
0.40204405347090477 False
=======================
=======================
OpenFace, Manual cosine distance comparison took:0.619 seconds
0.13767012465665296
OpenFace, DeepFace verify image comparison took:0.139 seconds
0.13767012465665296 False
=======================
=======================
DeepID, Manual cosine distance comparison took:0.128 seconds
0.023040547736539407
DeepID, DeepFace verify image comparison took:0.075 seconds
0.023040547736539407 False
=======================
=======================
ArcFace, Manual cosine distance comparison took:1.029 seconds
0.5930208328382623
ArcFace, DeepFace verify image comparison took:0.242 seconds
0.5930208328382623 True ✅
=======================
=======================
Dlib, Manual cosine distance comparison took:0.186 seconds
0.06620544503724635
Dlib, DeepFace verify image comparison took:0.090 seconds
0.06620544503724635 True
=======================
=======================
SFace, Manual cosine distance comparison took:0.155 seconds
0.5371376296136141
SFace, DeepFace verify image comparison took:0.089 seconds
0.5371376296136141 True ✅
=======================
```

---

## 🧠 Model Selection Recommendations

| Scenario                             | Recommended Model  |
| ------------------------------------ | ------------------ |
| Best accuracy for face ID            | `ArcFace`          |
| Fastest for edge/mobile use          | `Facenet`, `SFace` |
| Lightweight & classic academic model | `OpenFace`         |
| Simple and well-known (older)        | `VGG-Face`         |
| C++ integration / legacy systems     | `Dlib`             |

---

## ⚒️ Dlib Setup for macOS (optional)

```bash
brew install cmake
xcode-select --install
brew install boost
pip install dlib
```

---

## ✅ Summary

- 📦 Use DeepFace to compare face images using modern deep learning models
- 🔍 Explore 8 supported face recognition backends
- 📊 Benchmark your own face data to choose the best model
- 🧠 Support direct embedding comparisons for fast pipelines

---

## 👨‍💻 Author

**Emiliano Roberti**  
_Machine Learning and AI Developer_

For questions, optimisations or contributions, feel free to fork the repo or get in touch.
