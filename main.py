from deepface import DeepFace

def verify():
  """
  function to compare two faces and work out if there is a match based on a threashold
  """
  print("starting face comparison")
  result = DeepFace.verify(
    img1_path='./images/face1.png',
    img2_path='./images/face3.png'
  )
  print("completed face comparison")
  print(result)

def find(img_path, db_path):
  """
  function to look for a face inside a file system folder structure
  """
  dfs = DeepFace.find(img_path=img_path, db_path=db_path)
  print(dfs)

def represent(img_path):
  embedding_objs = DeepFace.represent(img_path=img_path)
  for embedding_obj in embedding_objs:
    embedding = embedding_obj["embedding"]
    print(embedding)


#find("./images/face1.png", "my_db")
represent("./images/face1.png")