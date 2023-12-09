import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet3D(nn.Module):
    def __init__(self, block, layers, block_inplanes, n_input_channels=3, conv1_t_size=7, conv1_t_stride=1, no_max_pool=False, shortcut_type='B', widen_factor=1.0, num_output=3,):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels, self.in_planes, kernel_size=(conv1_t_size, 7, 7), stride=(
            conv1_t_stride, 2, 2), padding=(conv1_t_size // 2, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, block_inplanes[0], layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, block_inplanes[1], layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, block_inplanes[2], layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(
            block, block_inplanes[3], layers[3], shortcut_type, stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, num_output)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes *
                              block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(in_planes=self.in_planes, planes=planes,
                      stride=stride, downsample=downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def preprocess(self, input: torch.Tensor, last_step_output: torch.Tensor = None):
        """
        :param input: torch.Tensor, shape=(32, 32, 32)
        :param last_step_input: torch.Tensor, shape=(, 3)
        :return: torch.Tensor, shape=(3, 32, 32, 32)
        """
        origin_voxel = torch.clone(input)
        completed_voxel = origin_voxel - input
        last_step_voxel = torch.zeros((32, 32, 32))
        if last_step_output is not None:
            last_step_voxel[last_step_output[0],
                            last_step_output[1], last_step_output[2]] = -1
        mutil_channel_input = torch.stack(
            (input, completed_voxel, last_step_voxel), dim=0)

        return mutil_channel_input.unsqueeze(0)

    def inference(self, input: torch.Tensor, max_step: int):
        """
        :param input: torch.Tensor, shape=(32, 32, 32)
        :param max_step: int
        :return: list[torch.Tensor], torch.Tensor.shape=(, 3)
        """
        num_step = 0
        sequence = []
        with torch.no_grad():
            # classification
            # NOTE: mask shape should be tested
            mask = torch.zeros(32*32*32)
            mask[input.flatten().bool()] = 1
            mask = torch.cat((mask, torch.tensor([0])), dim=0)
            num_voxel = mask.sum().item()
            # print(mask)
            while num_step < num_voxel and num_step < max_step:
                last_step_output = sequence[-1] if num_step > 0 else None
                mutil_channel_input = self.preprocess(input, last_step_output)
                output = self(mutil_channel_input).squeeze(0)
                # print(output.shape)
                # output = output.round().long().squeeze(0)
                # if (output == -1).all():
                #     break
                
                # classifcation
                output = mask * F.softmax(output, dim=-1)
                pred = output.argmax(dim=-1).long().squeeze(0)
                if pred == 32*32*32:
                    break
                mask[pred] = 0
                position = self.label2position(pred)
                # print(position.shape)
                sequence.append(position)
                num_step += 1
                input[position[0], position[1], position[2]] = 0
        return sequence
    
    def label2position(self, label: torch.Tensor):
        """
        :param label: torch.Tensor, shape=(, 1)
        :return: torch.Tensor, shape=(, 3)
        """
        return torch.stack((label // (32 * 32), (label // 32) % 32, label % 32), dim=-1).long()
    

if __name__ == '__main__':
    cnn = ResNet3D(BasicBlock, [2, 2, 2, 2], [64, 128, 256, 512], num_output=32769)
    cnn.eval()

    input = torch.randint(0, 2, (32, 32, 32)).float()
    output = cnn.inference(input, 2)
    print(output)
