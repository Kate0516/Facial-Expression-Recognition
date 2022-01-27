import os,base64

from django.http import HttpResponse
from django.shortcuts import render
from django.template import loader

from .face_rec import photo_rec

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR =os.path.join(os.path.join(BASE_DIR, 'hci'),"uploadfiles")

crlt = ""
cshowimg = ""
def home(request):
    return render(request, 'home.html')
def photo(request):
    context = {}
    context['subtitle'] = 'PHOTO EMOTION RECOGNITION'
    context['showimg'] = 'images/smile.jpg'
    if request.method == 'GET':
        return render(request, 'photo.html',context)
    if request.POST:
        myfile = request.FILES.get('face')
        #把图片存储到服务器
        showpath,filepath = photo_rec.save_img(myfile)
        context['showimg'] = showpath
        #把存储好的图片输入模型训练，返回一个代表表情的字符串
        context['rlt'] = photo_rec.rec(filepath)
    return render(request, "photo.html", context)
def jump(request):
    context = {}
    context['jump'] = 'Hello World!'
    return render(request, "jump.html", context)
def emoji(request):
    context = {}
    context['subtitle'] = 'GET YOUR EMOJI'
    context['showimg'] = 'images/smile.jpg'
    context['emoji'] = 'images/happyemo.png'
    if request.method == 'GET':
        return render(request, 'emoji.html', context)
    if request.POST:
        #图片在请求的File属性里，是一个uploadfile
        myfile = request.FILES.get('face')
        #print(request.FILES.get('face'))
        showpath, filepath = photo_rec.save_img(myfile)
        context['showimg'] = showpath
        context['emoji'] = photo_rec.rec_and_emoji(filepath)
    return render(request, "emoji.html", context)
def camera(request):
    context = {}
    context['subtitle'] = 'USE CAMERA TO CAPTURE'
    context['showimg'] = 'images/happy.jpg'
    context['emoji'] = 'images/happyemo.png'
    if request.method == 'GET':
        return render(request, 'camera.html', context)
    if request.POST:
        #除去了前面data:image/png;base64,从base64处开始处理
        imgdata = base64.b64decode(request.POST.get('face')[22:])
        #这个地方不是预设好的POST，是自己写的POST，没有FILE，在JS中文件放在POST里发送的
        spath, fpath = photo_rec.save_img_camera(imgdata)#将base64存储为jpg，js里输出的是jpg，可更改
        print(spath)
        print(fpath)
    return render(request, 'camera.html', context)
def result(request):
    context = {}
    context['subtitle'] = 'RESULT'
    context['showimg'] = 'images/happy.jpg'
    context['emoji'] = 'images/happyemo.png'
    if request.method == 'GET':
        return render(request, 'camera.html',context)
    if request.POST:
        fpath = os.path.join(IMG_DIR,"capture.jpg")
        print(fpath)
        #fpath = repr(filepath)
        #print(fpath)
        context['rlt'] = photo_rec.rec(fpath)
        context['emoji'] = photo_rec.rec_and_emoji(fpath)
        #context['showimg'] = photo_rec.rec_and_emoji(fpath)
    return render(request, "camera.html", context)