import torch.nn as nn
import torch.nn.functional as F
class NET(nn.Module):
    def __init__(self):
        super(NET,self).__init__()
        self.conv1 = nn.Conv2d(6,64,7,stride=2)
        self.conv2 = nn.Conv2d(64,128,5,stride=2)
        self.conv3 = nn.Conv2d(128,256,5,stride=2)
        self.conv4 = nn.Conv2d(256, 256, 3, stride=1)
        self.conv5 = nn.Conv2d(256, 512, 3, stride=2)
        self.conv6 = nn.Conv2d(512, 512, 3, stride=1)
        self.conv7 = nn.Conv2d(512, 512, 3, stride=2)
        self.conv8 = nn.Conv2d(512, 512, 3, stride=1)
        self.conv9 = nn.Conv2d(512, 1024, 3, stride=2)

        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(8192,4096)
        self.fc2 = nn.Linear(4096,1024)
        self.fc3 = nn.Linear(1024,128)
        self.fc4 = nn.Linear(128,12)


    def forward(self,x):
        # x = torch.cat((x1,x2),1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))

        x = self.pool(x)
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x