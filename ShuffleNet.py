import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional
from torch.autograd import Variable
from torch.nn import init
import math

class ShuffleNet_Unit(nn.Module):

    def __init__(self, input_channels, output_channels, groups=3, withstride=1, stage=2):
        super(ShuffleNet_Unit, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.groups = groups
        self.withstride = withstride
        self.stage = stage
        self.bottleneck_channels = self.output_channels // 4

        # Get type of this unit (Add or Concat)
        self.last_layer_type = self.add
        self.get_type()

        # Get first layer group (for stage 2 we don't need group convolution)
        self.GConv_1x1_group = self.get_first_layer_group()

        # First layer
        self.GConv_1x1 = self.construct_gconv_1x1(self.input_channels, self.bottleneck_channels, self.GConv_1x1_group, batch_norm=True, relu=True)

        # Second layer
        self.DWConv_3x3 = nn.Conv2d(self.bottleneck_channels, self.bottleneck_channels, kernel_size=3, padding=1, stride=self.withstride, groups=self.bottleneck_channels)
        self.DW_BN = nn.BatchNorm2d(self.bottleneck_channels)

        # Third layer
        self.GConv_1x1_last = self.construct_gconv_1x1(self.bottleneck_channels, self.output_channels, self.groups, batch_norm=True, relu=False)

    def forward(self, x):
        if self.withstride == 1:
            shortcut_path = x
        else:
            shortcut_path = nn.functional.avg_pool2d(x, kernel_size=3, stride=2, padding=1)
        output = self.GConv_1x1(x)
        #print("1",output.size())
        output = self.channel_shuffle(output, self.groups)
        output = self.DWConv_3x3(output)
        output = self.DW_BN(output)
        #print("2",output.size())
        output = self.GConv_1x1_last(output)
        output = self.last_layer_type(shortcut_path, output)
        #print("3",output.size())
        output = nn.functional.relu(output)
        return output

    def get_type(self):
        if self.withstride == 2:
            self.output_channels = self.output_channels - self.input_channels
            self.last_layer_type = self.concat
        else:
            self.last_layer_type = self.add

    @staticmethod
    def add(x, y):
        return x + y

    @staticmethod
    def concat(x, y):
        return torch.cat((x, y), 1)

    def get_first_layer_group(self):
        if self.stage <= 2:
            return 1
        else:
            return self.groups

    def construct_gconv_1x1(self, input_channels, output_channels, groups, batch_norm=True, relu=False):
        modules = OrderedDict()
        conv = nn.Conv2d(input_channels, output_channels, kernel_size=1, groups=groups, stride=1)
        modules['1x1_GConv'] = conv

        if batch_norm:
            modules['batch_norm'] = nn.BatchNorm2d(output_channels)
        if relu:
            modules['relu'] = nn.ReLU()
        if len(modules) > 1:
            return nn.Sequential(modules)
        else:
            return conv

    @staticmethod
    def channel_shuffle(x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = x.transpose(1, 2)
        # out of memory
        x = x.reshape(batchsize, -1, height, width)
        return x


class ShuffleNet(nn.Module):

    def __init__(self, groups=3, input_channels=3, num_classes=1000, ratio=1):
        # Set initial parameters
        super(ShuffleNet, self).__init__()
        self.groups = groups
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.ratio = ratio
        self.output_channel_dim = self.get_output_channel_dim()

        # Stage 1
        self.stage1 = self.construct_stage(1)

        # Stage 2
        self.stage2 = self.construct_stage(2)

        # Stage 3
        self.stage3 = self.construct_stage(3)

        # Stage 4
        self.stage4 = self.construct_stage(4)

        # FC Layer
        self.full_connected = nn.Linear(self.output_channel_dim[-1], self.num_classes)

    def forward(self, x):
        output = self.stage1(x)
        output = self.stage2(output)
        output = self.stage3(output)
        output = self.stage4(output)
        output = nn.functional.avg_pool2d(output, output.data.size()[-2:])
        output = output.view(output.size(0), -1)
        output = self.full_connected(output)
        output = nn.functional.log_softmax(output, dim=1)
        return output

    def construct_stage(self, stage):
        modules = OrderedDict()
        if stage == 1:
            stage_name = "Stage{}".format(stage)
            modules[stage_name + "_1"] = nn.Conv2d(self.input_channels, self.output_channel_dim[0], kernel_size=3, stride=2, padding=1)
            modules[stage_name + "_2"] = nn.BatchNorm2d(self.output_channel_dim[0])
            modules[stage_name + "_3"] = nn.ReLU()
            modules[stage_name + "_4"] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        elif stage == 2 or stage == 3 or stage == 4:
            stage_name = "Stage{}".format(stage)

            # Add the first Concat unit with stride = 2
            modules[stage_name + "_1"] = ShuffleNet_Unit(self.output_channel_dim[stage-2], self.output_channel_dim[stage-1],groups=self.groups, withstride=2, stage=stage)

            # Add repeated units with stride = 1
            repeat_num = self.get_repeat_num(stage)
            for i in range(repeat_num):
                module_name = stage_name + "_{}".format(i + 2)
                modules[module_name] = ShuffleNet_Unit(self.output_channel_dim[stage-1],self.output_channel_dim[stage-1],groups=self.groups, withstride=1, stage=stage)

        else:
            print("Wrong stage number! Valid stage number should be 1,2,3 or 4!")
            exit()
        return nn.Sequential(modules)

    def get_output_channel_dim(self):
        if self.groups == 1:
            dim = [24, 144, 288, 576]
        elif self.groups == 2:
            dim = [24, 200, 400, 800]
        elif self.groups == 3:
            dim = [24, 240, 480, 960]
        elif self.groups == 4:
            dim = [24, 272, 544, 1088]
        elif self.groups == 8:
            dim = [24, 384, 768, 1536]
        else:
            print("Wrong group number! Valid group number should be 1,2,3,4 or 8!")
            exit()

        dim[0] = int(math.ceil(dim[0] * self.ratio / self.groups) * self.groups)
        g = self.groups * 4
        for k in range(1,4):
            dim[k] = int(math.ceil(dim[k] * self.ratio / g) * g)

        return dim

    @staticmethod
    def get_repeat_num(stage):
        if stage == 2 or stage == 4:
            return 3
        elif stage == 3:
            return 7
        else:
            print("Wrong stage number! Valid stage number should be 2, 3 or 4!")
            exit()


def shufflenetv1(groups=3, ratio=1.0):
    model = ShuffleNet(groups=groups, ratio=ratio)
    return model


if __name__ == "__main__":
    # model = ShuffleNet()
    # print(model)
    torch.set_default_tensor_type("torch.DoubleTensor")
    print("Loading data...")






