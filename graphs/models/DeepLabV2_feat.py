import sys
import os
sys.path.append(os.path.abspath('.'))
from graphs.models.DeepLabV2 import *

class DeeplabResnetFeat(DeeplabResnet):

    def forward(self, x):
        input_size = x.size()[2:]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)
        feat = x
        x1 = self.layer6(x)  # classifier module
        x1 = F.interpolate(x1, size=input_size, mode='bilinear', align_corners=True)

        return x1, feat


class DeeplabVGGFeat(DeeplabVGG):

    def forward(self, x):

        input_size = x.size()[2:]
        x = self.features(x)
        feat = x
        x1 = self.classifier(x)
        x1 = F.interpolate(x1, size=input_size, mode='bilinear', align_corners=True)

        return x1, feat


def DeeplabFeat(num_classes, backbone, pretrained=True):
    print('DeeplabV2 is being used with {} as backbone'.format(backbone))
    if backbone == 'ResNet101':
        model = DeeplabResnetFeat(Bottleneck, [3, 4, 23, 3], num_classes)
        if pretrained:
            restore_from = './pretrained_model/DeepLab_resnet_pretrained_init-f81d91e8.pth'
            saved_state_dict = torch.load(restore_from)

            new_params = model.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split('.')
                if not i_parts[1] == 'layer5':
                    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
            model.load_state_dict(new_params)
    elif backbone == 'VGG16':
        restore_from = './pretrained_model/vgg16-397923af.pth'
        model = DeeplabVGGFeat(num_classes, restore_from=restore_from, pretrained=pretrained)
    else:
        raise Exception

    return model


