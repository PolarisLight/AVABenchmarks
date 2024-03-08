import torch
import torchvision
import torch.nn as nn
import timm
import torch.nn.functional as F


def conv2d_bn(in_planes, out_planes, kernel_size=1, stride=1, padding=0):
    "3x3 convolution with padding and batch norm"
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, size=768, name=''):
        super(InceptionBlock, self).__init__()
        self.branch1x1 = conv2d_bn(in_channels, size, 1, 1)

        self.branch3x3_1x1 = conv2d_bn(in_channels, size, 1, 1)
        self.branch3x3_1x3 = conv2d_bn(size, size // 2, (1, 3), padding=(0, 1))
        self.branch3x3_3x1 = conv2d_bn(size, size // 2, (3, 1), padding=(1, 0))

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            conv2d_bn(in_channels, size, 1, 1)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1x1(x)
        branch3x3 = torch.cat([
            self.branch3x3_1x3(branch3x3),
            self.branch3x3_3x1(branch3x3),
        ], 1)

        branch_pool = self.branch_pool(x)

        outputs = torch.cat([branch1x1, branch3x3, branch_pool], 1)
        return outputs


class InceptionResNetV2Pooled(nn.Module):
    def __init__(self, input_shape=(3, 299, 299), pool_size=(5, 5), return_sizes=False):
        super(InceptionResNetV2Pooled, self).__init__()
        self.return_sizes = return_sizes
        # 加载预训练的InceptionResNetV2模型
        self.model_base = timm.create_model('inception_resnet_v2', pretrained=True, features_only=True)
        self.pool_size = pool_size
        self.mixed_features = []
        self.pooling = nn.AdaptiveAvgPool2d(self.pool_size)
        # register hook for all layers that has 'mixed' in their name, get the output of the layer
        self.model_base.mixed_5b.register_forward_hook(self.get_features_hook)
        self.model_base.mixed_6a.register_forward_hook(self.get_features_hook)
        self.model_base.mixed_7a.register_forward_hook(self.get_features_hook)
        self.feature_layers = []
        for name, layer in self.model_base.named_children():
            for subname, sublayer in layer.named_children():
                print(name, subname)
                if 'repeat' in name:
                    sublayer.register_forward_hook(self.get_features_hook)
                    self.feature_layers.append(name+' '+subname)

    def get_features_hook(self, module, input, output):
        self.mixed_features.append(output)

    def forward(self, x):
        print(self.feature_layers)
        # 使用timm库的forward_features方法直接获取特征图
        x = self.model_base(x)

        # 调整特征图的大小
        for i, feature in enumerate(self.mixed_features):
            self.mixed_features[i] = self.pooling(feature)
            # print(self.mixed_features[name].shape)
        for i in self.mixed_features:
            print(i.shape)

        mixed_features = torch.concatenate([self.mixed_features[name] for name in self.mixed_features], 1)
        print(mixed_features.shape)


def main():
    model = InceptionResNetV2Pooled().cuda()
    # for name, param in model.model_base.named_parameters():
    #     print(name, param.shape)
    img = torch.rand(1, 3, 299, 299).cuda()
    out = model(img)


if __name__ == "__main__":
    main()
