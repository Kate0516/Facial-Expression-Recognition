import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os

train_path = './data/train/'
vaild_path = './data/val/'
data_path = './fer2013/fer2013.csv'

def make_dir():
    for i in range(0,7):
        p1 = os.path.join(train_path,str(i))
        p2 = os.path.join(vaild_path,str(i))
        if not os.path.exists(p1):
            os.makedirs(p1)
        if not os.path.exists(p2):
            os.makedirs(p2)

def save_images():
    df = pd.read_csv(data_path)
    t_n = [1 for i in range(0,7)]
    v_n = [1 for i in range(0,7)]
    for index in range(len(df)):
        emotion = df.loc[index][0] #int represent the emotion
        image = df.loc[index][1] #a string of number represent the pixel of the image
        usage = df.loc[index][2] # is training or valid
        data_array = list(map(float, image.split())) #list
        data_array = np.asarray(data_array)
        image = data_array.reshape(48, 48)
        im = Image.fromarray(image).convert('L') #8bit grey picture
        if(usage=='Training'):
            t_p = os.path.join(train_path,str(emotion),'{}.jpg'.format(t_n[emotion]))
            im.save(t_p)
            t_n[emotion] += 1
        else:
            v_p = os.path.join(vaild_path,str(emotion),'{}.jpg'.format(v_n[emotion]))
            im.save(v_p)
            v_n[emotion] += 1
    return t_n, v_n

def show_statistics(t_n, v_n):
    classes = ['angry','disgust','fear','happy','sad','surprised','normal']
    plt.subplot(2,  1,  1)
    plt.bar(classes, t_n, color =  'g', align =  'center')
    plt.title('train')
    plt.subplot(2,  1,  2)
    plt.bar(classes, v_n, color =  'b', align =  'center')
    plt.title('valid')
    plt.show()

make_dir()
t_n, v_n = save_images()
print(t_n,v_n)
show_statistics(t_n,v_n)



