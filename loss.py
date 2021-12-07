import torch.nn as nn
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
import torch
global batch_size

batch_size = 20
use_gpu = torch.cuda.is_available()

#损失函数和优化方法
class My_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, labels):
        label = []
        for c in range(0, batch_size-1):
            label_tmp = []
            for j in range(0, 12):
                label_tmp.append(labels[j][c].detach().numpy())
            label.append(label_tmp)
        label = np.array(label)
        label = torch.from_numpy(label)
        if(use_gpu):
            label=label.cuda()
        # for i in range(0,batch_size):
        loss = outputs-label

            # label_t = label[i]
            # outputs = outputs[i]
            # # labels = labels[i]
            # outputs = outputs.reshape(3,4)
            # # label_t = np.array(label_t)
            # label_t = label_t.reshape(3,4)
            # x1 = outputs[0:3,0:3]
            # x2 = outputs[2,2]
            # x2.reshape(1,-1)
            # x2 = np.array(x2)
            #
            # y1 = label_t[0:3,0:3]
            # y2 = label_t[2,2]
            # y2.reshape(1,-1)
            # y2 = np.array(y2)
            # #旋转矩阵转旋转角
            # R1 = list(Quaternion(matrix=x1))
            # R1 = np.array(R1)
            # R2 = list(Quaternion(matrix=y1))
            # R2 = np.array(R2)#旋转矩阵转欧拉角
            # #欧拉角转旋转角
            # R1 = R.from_quat(R1).as_euler('zyx')
            # R2 = R.from_quat(R2).as_euler('zyx')
            # R1 = np.array(R1)
            # R2 = np.array(R2)
            # #损失
            # loss = pow((np.linalg.norm((R2-R1),ord=2)),2) + pow((np.linalg.norm((y2-x2),ord=2)),2)#旋转角与真实值的差值二范数平方加坐标与真实值差值二范数的平方
            # loss += loss
        # loss = loss/batch_size
        loss = loss.cpu().detach().numpy()
        loss = np.sum(loss).T
        loss = torch.tensor(loss)
        loss.requires_grad_(True)
        return loss