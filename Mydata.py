from loader import MyLoader
from torch.utils.data import Dataset, DataLoader
class MyDataset(Dataset):
    # 构造函数设置默认参数
    def __init__(self, txt1, transform=None, target_transform=None, loader=MyLoader):
        imgs=[]
        imgg=[]
        with open(txt1, 'r+') as f:
            for line in f:
                line = line.strip('\n')  # 移除字符串首尾的换行符
                line = line.rstrip()  # 删除末尾空
                words = line.split(' ')  # 以空格为分隔符 将字符串分成
                for i in range(1,17):
                    imgg.append(float(words[i]))
                imgs.append((words[0], imgg))
                imgg=[]
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        # 调用定义的loader方法
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    def __len__(self):
        return len(self.imgs)
