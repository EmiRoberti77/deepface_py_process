from deepface import DeepFace

def verify():
  print("starting face comparison")
  result = DeepFace.verify(
    img1_path='./images/face1.png',
    img2_path='./images/face3.png'
  )
  print("completed face comparison")
  print(result)

def find(img_path, db_path):
  dfs = DeepFace.find(img_path=img_path, db_path=db_path)
  print(dfs)

 
find("./images/face1.png", "my_db")