import os
import torch
import torchvision
import cv2
from torchvision import transforms,datasets,models
from PIL import Image
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOW_DIR = os.path.join(os.path.join(os.path.dirname(BASE_DIR),"statics"),"images")
IMG_DIR=os.path.join(BASE_DIR, 'uploadfiles')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
def save_img(file):
    filepath = filepath = os.path.join(IMG_DIR, file.name)
    showpath = os.path.join(SHOW_DIR, file.name)
    with open(filepath, 'wb') as f:
         for i in file.chunks():  # 分包写入
             f.write(i)
    with open(showpath, 'wb') as f:
         for i in file.chunks():  # 分包写入
             f.write(i)
    print('upload file save at: ',filepath)
    return showpath,filepath
def save_img_camera(imgdata):
    filepath = filepath = os.path.join(IMG_DIR, "capture.jpg")
    showpath = os.path.join(SHOW_DIR, "capture.jpg")
    with open(filepath, 'wb') as f:
        f.write(imgdata)
    with open(showpath, 'wb') as f:
        f.write(imgdata)
    print('upload file save at: ',filepath)
    return showpath,filepath

def pre_process(filepath):
    img = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = Image.fromarray(img)
    trans = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(),
        torchvision.transforms.Resize((48, 48)),
        torchvision.transforms.ToTensor(),
        transforms.Normalize([0.485, ], [0.229, ])
    ])  # 定义图像变换以符合网络输入
    img = trans(img)
    img = img.reshape(1, 1, 48, 48)
    return img

def rec(filepath):
    modelpath = os.path.join(MODEL_DIR, 'resnet0.pth')
    model = torch.load(modelpath)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
    else:
        print("cuda is not avaliable")
    img = pre_process(filepath).cuda()

    emotion = ["angry", "disgust", "fear", "happy", "sad", "surprised", "normal"]  # 表情标签
    res = model(img).max(1)[1].item()
    emo = emotion[res]
    return emo
def rec_and_emoji(filepath):
    emo = rec(filepath)
    emo += 'emo.png'
    showpath = os.path.join(SHOW_DIR, emo)
    return showpath
# filepath = r'D:\SCHOOL\HCI\Code\hci\hci\uploadfiles\test.jpg'
# img = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img,(48,48))
# img = img.reshape(1,1,48,48)
# pre = model(img).max(1)[1].item()
# print(filepath)
# print(img)
# print(type(img))
#print(SHOW_DIR)
