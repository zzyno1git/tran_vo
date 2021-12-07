import torch.optim as optim
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from Mydata import MyDataset
from Net import NET
from loss import My_loss,batch_size
#数据加载
root = r"/home/heu612160/PycharmProjects/ZZYPytorch/txt" #00rr.txt绝对位置
train_data = MyDataset(txt1=root + '/' + '00rr.txt', transform=transforms.ToTensor())
# train_data 和test_data包含多有的训练与测试数据，调用DataLoader批量加载
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True,drop_last=True)
net = NET()
#model load
# checkpoint = torch.load('./checkpoint/05checkpoint/05secKITTI_epoch_30.ckpt')
# net.load_state_dict(checkpoint['net'])
net = net.train()
criterion = My_loss()
optimizer = optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
use_gpu = torch.cuda.is_available()
if (use_gpu):
    print('gpu is available.')
    net = net.cuda()
    criterion = criterion.cuda()
#训练
for epoch in range(10):
    print("epoch %d" %epoch)
    for batch_idx, data in enumerate(train_loader):
        inputs1, labels = data
        # for q in range(0, batch_size-1):
        #     # tmp_x = torch.cat((torch.unsqueeze(inputs1[0], dim=0), torch.unsqueeze(inputs1[1], dim=0)), 1)
        #     # x = torch.cat((x, tmp_x), 0)
        #     x = torch.cat((torch.unsqueeze(inputs1[q], dim=0), torch.unsqueeze(inputs1[q + 1], dim=0)), 1)
        if(use_gpu):
            x = inputs1.cuda()
        inputs = x
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs,labels)
        loss.backward(loss.clone().detach())
        optimizer.step()
        state = {
            'net':net.state_dict(),
            'epoch':epoch + 1,
        }
        print(epoch,batch_idx)
        if (epoch+1) % 5 ==0:
            torch.save(state,'./checkpoint/00checkpoint/00KITTI_epoch_%d.ckpt' %(epoch+1))
print('Finished Training')
