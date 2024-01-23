import torch.nn as nn
import torch


class VGG(nn.Module):
    def __init__(self, feature, num_class=1000, init_weight=False):
        """
        :param feature: 特征提取网络
        :param num_class: 识别类别
        :param init_weight: 是否初始权重
        """
        super(VGG, self).__init__()
        self.feature = feature
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),  # 随机释放50%的神经元防止过拟合
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),  # 随机释放50%的神经元防止过拟合
            nn.Linear(4096, num_class)
        )
        if init_weight:
            self.init_weights()

    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(x, start_dim=1)  # 从第几维开始展开
        x = self.classifier(x)
        return x

    def init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


def extract_features(cfg: list):
    """创建特征提取网络"""
    layers = []
    in_channels = 3
    for i in cfg:
        if i == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels, i, kernel_size=3, stride=1, padding=1)]
            layers += [nn.ReLU(True)]
            in_channels = i
    return nn.Sequential(*layers)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
              'M'],
}


def vgg(model_name="vgg16", **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]
    feature = extract_features(cfg)
    model = VGG(feature, **kwargs)
    return model


vgg()
