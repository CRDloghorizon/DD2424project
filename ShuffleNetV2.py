# ShuffleNetV2 based on
# ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design

import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional
from torch.autograd import Variable
from torch.nn import init
import math

class ShuffleNetV2_Unit(nn.Module):
    
    def __init__(self, input_channels, output_channels, withstride):
        super(ShuffleNetV2_Unit, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.withstride = withstride # withstride = [1,2] unit type == DWconv stride
        branch_channels = self.output_channels // 2

        # withstride 1,
        if self.withstride == 1:
            self.right = nn.Sequential(
                self.conv_1x1_sh(branch_channels, branch_channels),
                self.dwconv_3x3_sh(branch_channels, branch_channels, self.withstride),
                self.conv_1x1_sh(branch_channels, branch_channels)
            )
        else:
            self.left = nn.Sequential(
                self.dwconv_3x3_sh(self.input_channels, self.input_channels, self.withstride),
                self.conv_1x1_sh(self.input_channels, branch_channels)
            )
            self.right = nn.Sequential(
                self.conv_1x1_sh(self.input_channels, branch_channels),
                self.dwconv_3x3_sh(branch_channels, branch_channels, self.withstride),
                self.conv_1x1_sh(branch_channels, branch_channels)
            )

    def forward(self, x):
        if self.withstride == 1:
            # channel split into half
            x1 = x[:, :x.shape[1]//2, :, :]
            x2 = x[:, x.shape[1]//2:, :, :]
            output = self.concat(x1, self.right(x2))
        elif self.withstride == 2:
            output = self.concat(self.left(x), self.right(x))
        else:
            print("Wrong with stride, should be 1 or 2.")
            exit()

        output = self.channel_shuffle(output, 2)

        return output

    @staticmethod
    def concat(x, y):
        return torch.cat((x, y), 1)

    # convolution 1x1 BN ReLU
    def conv_1x1_sh(self, input_channels, output_channels):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )

    # depthwise convolution 3x3 BN
    def dwconv_3x3_sh(self, input_channels, output_channels, stride):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, groups=output_channels, stride=stride, padding=1),
            nn.BatchNorm2d(output_channels)
        )

    @staticmethod
    def channel_shuffle(x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = x.transpose(1, 2)
        x = x.reshape(batchsize, -1, height, width)
        return x


class ShuffleNetV2(nn.Module):
    def __init__(self, input_channels=3, num_classes=1000, ratio=1.0, input_size=224):
        # Set initial parameters
        super(ShuffleNetV2, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.ratio = ratio
        self.input_size = input_size

        self.repeat_num = [3, 7, 3]
        if ratio == 0.5:
            self.out_channel = [24, 48, 96, 192, 1024]
        elif ratio == 1.0:
            self.out_channel = [24, 116, 232, 464, 1024]
        elif ratio == 1.5:
            self.out_channel = [24, 176, 252, 704, 1024]
        elif ratio == 2.0:
            self.out_channel = [24, 244, 488, 976, 2048]
        else:
            print("Wrong network ratio.")
            exit()

        # Stage 1 Conv + maxpool
        self.stage1 = self.construct_stage(1)

        # Stage 2
        self.stage2 = self.construct_stage(2)

        # Stage 3
        self.stage3 = self.construct_stage(3)

        # Stage 4
        self.stage4 = self.construct_stage(4)

        # Stage 5 Conv + globalpool
        self.stage5 = self.construct_stage(5)

        # FC
        self.classifier = nn.Sequential(
            nn.Linear(self.out_channel[-1], self.num_classes)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def construct_stage(self, stage):
        if stage == 1:
            return nn.Sequential(
                nn.Conv2d(self.input_channels, self.out_channel[0], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(self.out_channel[0]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        elif stage in [2, 3, 4]:
            modules = OrderedDict()
            stage_name = "Stage{}".format(stage)
            modules[stage_name + "_1"] = ShuffleNetV2_Unit(self.out_channel[stage-2], self.out_channel[stage-1], 2)
            for i in range(self.repeat_num[stage-2]):
                repeat_name = stage_name + "_{}".format(i + 2)
                modules[repeat_name] = ShuffleNetV2_Unit(self.out_channel[stage-1], self.out_channel[stage-1], 1)

            return nn.Sequential(modules)
        elif stage == 5:
            return nn.Sequential(
                nn.Conv2d(self.out_channel[stage-2], self.out_channel[-1], kernel_size=1, stride=1),
                nn.BatchNorm2d(self.out_channel[-1]),
                nn.ReLU(),
                nn.AvgPool2d(int(self.input_size/32))
            )
        else:
            print("Wrong stage number. 1 to 5.")
            exit()


def shufflenetv2(ratio=1.0):
    model = ShuffleNetV2(ratio=ratio)
    return model


if __name__ == "__main__":
    # testing
    # model = ShuffleNetV2()
    # print(model)
    torch.set_default_tensor_type("torch.DoubleTensor")
    print("Loading data...")
