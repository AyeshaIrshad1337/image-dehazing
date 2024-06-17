import os 

def data_loader(path):
    img=[]
    for img in path:
        img.append(os.path.join(path,img))
    return img
