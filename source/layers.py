import torch
import torch.nn as nn

class residual_block(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, stride=1):
        super(residual_block, self).__init__()
        self.stride = stride
        # Residual Block
        self.residual_block = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(),
            nn.Conv2d(mid_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim),
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.residual_block(x) # F(x)
        if self.stride == 1:
            out = out + x # F(x) + x
        out = self.relu(out)
        return out
    
class resNet18(nn.Module):
    def __init__(self, in_dim, out_dim, first_kernel=7, first_stride=2, stem_pooling=True):
        super(resNet18, self).__init__()
        self.stem_pooling = stem_pooling
        
        # stem
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=first_kernel, stride=first_stride, padding=1)
        self.pooling = nn.MaxPool2d(3, 2)
        # residual blocks
        self.conv2 = nn.Sequential(
            residual_block(64, 64, 64),
            residual_block(64, 64, 64),
        )
        self.conv3 = nn.Sequential(
            residual_block(64, 128, 128, 2),
            residual_block(128, 128, 128),
        )
        self.conv4 = nn.Sequential(
            residual_block(128, 256, 256, 2),
            residual_block(256, 256, 256),
        )
        self.conv5 = nn.Sequential(
            residual_block(256, 512, 512, 2),
            residual_block(512, 512, 512),
        )
        # head layers
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, out_dim)
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        x = self.conv1(x)
        if self.stem_pooling:
            x = self.pooling(x)
        # residual blocks
        f1 = self.conv2(x)
        f2 = self.conv3(f1)
        f3 = self.conv4(f2)
        f4 = self.conv5(f3)
        
        # head laters
        x = torch.flatten(self.gap(f4), 1)
        x = self.fc(x)
        y = self.softmax(x)
        return y, f1, f2, f3, f4
    
class loss_pred_module(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(loss_pred_module, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.lin = nn.Linear(in_dim, out_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = torch.flatten(self.gap(x), 1)
        x = self.lin(x)
        y = self.relu(x)
        return y      
        
class active_learning_model(nn.Module):
    def __init__(self, in_dim, out_dim, first_kernel=7, first_stride=2, stem_pooling=True, loss_mid_dim=128, K=1000, block_dims=[64, 128, 256, 512]):
        super(active_learning_model, self).__init__()
        # for cifar10 data
        self.base_model = resNet18(in_dim, out_dim, first_kernel, first_stride, stem_pooling)
        self.loss_pred_modules1 = loss_pred_module(block_dims[0], 1)
        self.loss_pred_modules2 = loss_pred_module(block_dims[1], 1)
        self.loss_pred_modules3 = loss_pred_module(block_dims[2], 1)
        self.loss_pred_modules4 = loss_pred_module(block_dims[3], 1)
        self.loss_fc = nn.Linear(4, 1)
    
    def forward(self, x):
        y, f1, f2, f3, f4 = self.base_model(x)
        # predict loss
        # features = [f1, f2, f3, f4]
        # features_hat = []
        # for i in range(len(features)):
        #     features_hat.append(self.loss_pred_modules[i](features[i]))
        f1_hat = self.loss_pred_modules1(f1)
        f2_hat = self.loss_pred_modules2(f2)
        f3_hat = self.loss_pred_modules3(f3)
        f4_hat = self.loss_pred_modules4(f4)
        f_cat = torch.cat([f1_hat, f2_hat, f3_hat, f4_hat], dim=-1)
        loss_pred = self.loss_fc(f_cat)
        return y, loss_pred
    
    def freeze_loss_predict(self, flag):
        for p in self.loss_pred_modules1.parameters():
            p.requires_grad = flag
        for p in self.loss_pred_modules2.parameters():
            p.requires_grad = flag
        for p in self.loss_pred_modules3.parameters():
            p.requires_grad = flag
        for p in self.loss_pred_modules4.parameters():
            p.requires_grad = flag
        
        for p in self.loss_fc.parameters():
            p.requires_grad = flag
        

          
                                
        
        