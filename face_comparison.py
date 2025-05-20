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

  