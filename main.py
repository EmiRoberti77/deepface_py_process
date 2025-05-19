from deepface import DeepFace
print("starting face comparison")
result = DeepFace.verify(
  img1_path='./images/face1.png',
  img2_path='./images/face3.png'
)
print("completed face comparison")
print(result)