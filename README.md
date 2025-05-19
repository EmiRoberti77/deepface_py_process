# 🧠 DeepFace Face Verification Example

This project demonstrates how to build and run a **Python DeepFace face verification** task comparing two images.

## 📂 Project Structure

```
deepface_sample/
├── images/
│   ├── face1.png
│   └── face2.png
├── main.py
└── requirements.txt
```

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

### `requirements.txt`

```
tensorflow==2.12.0
deepface==0.0.79
numpy
pandas
opencv-python
tf-keras
```

> ⚠️ **Note:** TensorFlow 2.19.0 requires `tf-keras`. If you use 2.19.0, also add `tf-keras` to the requirements.

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
pip install -r requirements.txt
```

_Or manually if using TensorFlow 2.19.0:_

```bash
pip install tensorflow==2.19.0 tf-keras deepface numpy pandas opencv-python
```

### 3. Run the Face Verification

Make sure you have two images in the `images/` folder named `face1.png` and `face2.png`.

```bash
python main.py
```

Expected output:

```
starting face comparison
completed face comparison
{ 'verified': True, 'distance': 0.23, ... }
```

---

## ✅ Summary

- ✅ Uses DeepFace for face verification
- ✅ Compares two images for face similarity
- ✅ Works in a Python virtual environment

---

## 🛠️ Troubleshooting

- **TensorFlow Compatibility:** Use Python 3.11 or lower.
- **Dependency Conflicts:** Prefer TensorFlow 2.12.0 for best compatibility.
- **Missing `tf-keras`:** Install `tf-keras` if using TensorFlow 2.19.0 or later.

## Author

`Emiliano Roberti`
