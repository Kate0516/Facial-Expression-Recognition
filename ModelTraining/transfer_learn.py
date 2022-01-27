import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets,models,transforms
import matplotlib.pyplot as plt
import time
import os
import copy

BATCH_SIZE=256
EPOCH=15
LR=0.001
DAMP_STEP=20 #after these step, the learning rate damping the amount
DAMP_RATE=0.1
data_dir = './data'

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(20)    # 暂停一会，以便更新绘图

if __name__ =="__main__":
    plt.ion()   # open interact mode so that the picture can be dynamic
    # 训练数据的扩充及标准化
    # 只进行标准化验证
    data_transforms = {
        'train': transforms.Compose([
            transforms.Grayscale(),  # 使用ImageFolder存图片时默认扩展为了三通道，现在变成一通道
            transforms.RandomHorizontalFlip(),  # 随机翻转
            transforms.ColorJitter(brightness=0.5, contrast=0.5),  # 随机调整亮度和对比度
            transforms.ToTensor(),
            transforms.Normalize([0.485,], [0.229,])
        ]),
        'val': transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,], [0.229,])
        ])
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(
        data_dir, x), data_transforms[x]) for x in ['train','val']}
    #num_workers is the number of thread using to load data
    dataloaders = {x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4) for x in ['train', 'val']}

    dataset_size = {x:len(image_datasets[x]) for x in ['train','val']}
    #class_name = image_datasets['train'].classes
    class_name = ['anger','disgust','fear','happy','sad','surprised','normal']
    print('dataset size: ',dataset_size)
    #print(class_name)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('using device: ', device)

    # 获取一批训练数据
    #inputs, classes = next(iter(dataloaders['train']))
    # 从批处理中生成网格
    #out = torchvision.utils.make_grid(inputs)
    #imshow(out, title='some image')


    # 参数schedule是来自torch.optim.lr_scheduler的LR调度对象
    def train_model(model, criterion, optimizer, schduler, num_epochs=EPOCH):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0  # 需要超过的准确率，这里设为了0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    schduler.step()
                    model.train()  # 训练模型
                else:
                    model.eval()  # 评估模型

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # 零化参数梯度
                    optimizer.zero_grad()

                    # 前向传递
                    # 如果只是训练的话，trace back
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # 训练时，反向传播 + 优化
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # 统计
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_size[phase]
                epoch_acc = running_corrects.double() / dataset_size[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                #拷贝模型
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()# whats this

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # 导入最优模型权重
        model.load_state_dict(best_model_wts)
        return model

    #展示预测效果
    def visualize_model(model, num_images=6):
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloaders['val']):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images // 2, 2, images_so_far)
                    ax.axis('off')
                    #ax.set_title('predicted: {} real: {}'.format(class_name[preds[j]],class_name[labels[j]]))
                    ax.set_title('predicted: {}'.format(class_name[preds[j]]))
                    imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return

            model.train(mode=was_training)


    model_ft = models.resnet18(pretrained=True) #use resnet
    #将resnet输入改成单通道
    model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 7) #修改全连接层

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    # 优化所有参数
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=LR, momentum=0.9)
    # 每7次，学习率衰减0.1
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer_ft, step_size=DAMP_STEP, gamma=DAMP_RATE)

    model_ft = train_model(model_ft, criterion, optimizer_ft,
                           exp_lr_scheduler, num_epochs=EPOCH)
    # 保存模型参数
    torch.save(model_ft, 'resnet0.pth')
    visualize_model(model_ft)
