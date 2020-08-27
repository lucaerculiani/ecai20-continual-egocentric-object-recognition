import torch

from functools import partial

import torchvision.models  as models
import torch.nn.functional as F



class ResNetEmbedding(torch.nn.Module):

    def __init__(self, base_resnet):
        super(ResNetEmbedding, self).__init__()

        self.base_resnet = base_resnet

    def forward(self, x):
        x = self.base_resnet.conv1(x)
        x = self.base_resnet.bn1(x)
        x = self.base_resnet.relu(x)
        x = self.base_resnet.maxpool(x)

        x = self.base_resnet.layer1(x)
        x = self.base_resnet.layer2(x)
        x = self.base_resnet.layer3(x)
        x = self.base_resnet.layer4(x)

        x = self.base_resnet.avgpool(x)

        return x

class InceptionEmbedding(torch.nn.Module):

    def __init__(self, base_network):
        super(InceptionEmbedding, self).__init__()

        self.base_network = base_network

    def forward(self, x):
        if self.base_network.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.base_network.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.base_network.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.base_network.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.base_network.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.base_network.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.base_network.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.base_network.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.base_network.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.base_network.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.base_network.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.base_network.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.base_network.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.base_network.Mixed_6e(x)
        # 17 x 17 x 768
        if self.base_network.training and self.base_network.aux_logits:
            aux = self.base_network.AuxLogits(x)
        # 17 x 17 x 768
        x = self.base_network.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.base_network.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.base_network.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=x.shape[-1])
        # 1 x 1 x 2048
        x = F.dropout(x, training=self.base_network.training)

def squeezenet1_1embedding(*args, **kwargs):
    return models.squeezenet1_1(*args, **kwargs).features


def resnet50embedding(*args, **kwargs):
    return ResNetEmbedding(models.resnet50(*args, **kwargs))

def resnet101embedding(*args, **kwargs):
    return ResNetEmbedding(models.resnet101(*args, **kwargs))


def resnet152embedding(*args, **kwargs):
    return ResNetEmbedding(models.resnet152(*args, **kwargs))

def inceptionv3embedding(*args, **kwargs):
    return InceptionEmbedding(models.inception_v3(*args, **kwargs))

def vggembedding(vgg_type, *args, **kwargs):
    vgg = vgg_type(*args, **kwargs)

    return vgg.features

def no_embedding(*args, **kwargs):
    return torch.nn.Sequential()


def get_embedding(string):
    if string not in _EMBEDDINGS.keys():
        e_str = "Valid embeddings: {}".format(list(_EMBEDDINGS.keys()))
        raise ValueError(e_str)
    return _EMBEDDINGS[string]


_EMBEDDINGS = { "squeezenet1_1": squeezenet1_1embedding,
               "resnet50" : resnet50embedding,
               "resnet101" : resnet101embedding,
               "resnet152" : resnet152embedding,
               "inceptionv3" : inceptionv3embedding,  
                "vgg11" : partial(vggembedding, models.vgg11),
                "vgg11_bn" : partial(vggembedding, models.vgg11_bn),
                "vgg13" : partial(vggembedding, models.vgg13),
                "vgg13_bn" : partial(vggembedding, models.vgg13_bn),
                "vgg16" : partial(vggembedding, models.vgg16),
                "vgg16_bn" : partial(vggembedding, models.vgg16_bn),
                "vgg19" : partial(vggembedding, models.vgg19),
                "vgg19_bn" : partial(vggembedding, models.vgg19_bn),
                "none" : no_embedding
               }

