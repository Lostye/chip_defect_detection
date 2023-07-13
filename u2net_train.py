import torch
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET
from model import U2NETP

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(reduction='mean')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
    loss0.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss6.item()))

    return loss0, loss


def main():
    # ------- 2. set the directory of training dataset --------
    model_name = 'u2netp'  # 'u2netp'
    # 数据集的路径
    data_dir = './data/data/'
    tra_image_dir = '/intel/train\\'
    tra_label_dir = '/intel_target\\'

    epoch_num = 100000
    # 训练的批次
    batch_size_train = 10

    tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*')
    tra_label_name_list = glob.glob(data_dir + tra_label_dir + '*')
    print(tra_img_name_list)
    print(tra_label_name_list)

    print("---")
    print("train images: ", len(tra_img_name_list))
    print("train labels: ", len(tra_label_name_list))
    print("---")

    train_num = len(tra_img_name_list)

    # 数据增样
    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_label_name_list,
        transform=transforms.Compose([
            RescaleT(320),
            RandomCrop(288),
            ToTensorLab(flag=0)]))
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True)

    # ------- 3. define model --------
    # define the net
    if model_name == 'u2net':
        net = U2NET(3, 1)
    elif (model_name == 'u2netp'):
        net = U2NETP(3, 1)

    if torch.cuda.is_available():
        net.cuda()
        print("CUDA")
    # 加载预训练权重
    model_dir = './saved_models/' + model_name + '/' + "u2netp.pth"
    if os.path.exists(model_dir):
        net.load_state_dict(torch.load(model_dir, map_location=device))
    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters())
    net.pool34.parameters()
    # ------- 5. training process --------
    print("---start training...")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    # save_frq改成自己的数据集总数
    save_frq = 174  # save the model every 174 iterations

    for epoch in range(0, epoch_num):
        net.train()

        for i, data in enumerate(salobj_dataloader):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            inputs, labels = data['image'], data['label']

            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                # A = nn.Parameter(inputs,requires_grad=True)

                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                            requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.item()
            running_tar_loss += loss2.item()

            # delete temporary outputs and loss
            # del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
                epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val,
                running_tar_loss / ite_num4val))

            # if ite_num % save_frq == 0:
            #     running_loss = 0.0
            #     running_tar_loss = 0.0
            #     net.train()  # resume train
            #     ite_num4val = 0
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()  # resume train
            ite_num4val = 1
        # 将训练的输出进行保存，保存的路径自己创建
        save_image(d0, "train_img/{}.png".format(epoch), nrow=4)
        torch.save(net.state_dict(), model_dir + model_name + "_bce_itr_%d_train_%3f_tar_%3f.pth" % (
            ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
        print("参数已保存！")


if __name__ == "__main__":
    main()
