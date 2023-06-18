import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from region_loss import RegionLossV2
from cfg import *
from dynamic_conv import dynamic_conv2d, ODConv2d
from pooling import GlobalMaxPool2d
from pooling import GlobalAvgPool2d
from pooling import Split
import pdb
from torchvision import transforms
import cv2
from torch.nn.parameter import Parameter
import math
from PMM import *
from graphModule import *
from torch.autograd import Variable
from math import exp
from yolov5_nb import *
from fusion_ import *
from GEblock import *
from PIL import Image
from prior import get_fft_feature
import os

def maybe_repeat(x1, x2):
    n1 = x1.size(0)
    n2 = x2.size(0)
    if n1 == n2:
        pass
    elif n1 < n2:
        assert n2 % n1 == 0
        shape = x1.shape[1:]
        nc = n2 // n1
        x1 = x1.repeat(nc, *[1] * x1.dim())
        x1 = x1.transpose(0, 1).contiguous()
        x1 = x1.view(-1, *shape)
    else:
        assert n1 % n2 == 0
        shape = x2.shape[1:]
        nc = n1 // n2
        x2 = x2.repeat(nc, *[1] * x2.dim())
        x2 = x2.transpose(0, 1).contiguous()
        x2 = x2.view(-1, *shape)
    return x1, x2


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class MaxPoolStride1(nn.Module):
    def __init__(self):
        super(MaxPoolStride1, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0, 1, 0, 1), mode='replicate'), 2, stride=1)
        return x


class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert (x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert (H % stride == 0)
        assert (W % stride == 0)
        ws = stride
        hs = stride
        x = x.view(B, C, H / hs, hs, W / ws, ws).transpose(3, 4).contiguous()
        x = x.view(B, C, H / hs * W / ws, hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, hs * ws, H / hs, W / ws).transpose(1, 2).contiguous()
        x = x.view(B, hs * ws * C, H / hs, W / ws)
        return x

# for route and shortcut
class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x

class Spectral_Discriminator(nn.Module):
    def __init__(self, height):
        super(Spectral_Discriminator, self).__init__()
        self.thresh = int(height / (2*math.sqrt(2)))
        self.linear = nn.Linear(self.thresh, 1)
    
    def forward(self, input: torch.Tensor):
        az_fft_feature = get_fft_feature(input)
        az_fft_feature[torch.isnan(az_fft_feature)] = 0
        
        return self.linear(az_fft_feature[:,-self.thresh:])

# support route shortcut and reorg
class Darknet(nn.Module):
    def __init__(self, darknet_file, learnet_file, train_ = True):
        super(Darknet, self).__init__()
        self.blocks = darknet_file if isinstance(darknet_file, list) else parse_cfg(darknet_file)
        self.learnet_blocks = learnet_file if isinstance(learnet_file, list) else parse_cfg(learnet_file)
        self.width = int(self.blocks[0]['width'])
        self.height = int(self.blocks[0]['height'])
        self.models, self.routs = self.create_network(self.blocks)  # merge conv, bn,leaky  ##查询Metafeature
        self.loss = self.models[len(self.models) - 1]
        self.train_ = train_
        
        if self.blocks[(len(self.blocks) - 1)]['type'] == 'region':
            self.anchors = self.loss.anchors
            self.num_anchors = self.loss.num_anchors
            self.num_classes = self.loss.num_classes

        self.header = torch.IntTensor([0, 0, 0, 0])
        self.seen = 0
        
        ###add
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(32)
        self.activate = nn.LeakyReLU(0.1, inplace=True)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=int((2 - 1) // 2))
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=int((2 - 1) // 2))
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=int((2 - 1) // 2))
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=int((2 - 1) // 2))
        
        self.conv5 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn5 = nn.BatchNorm2d(256)
        
        self.pool5_1 = GEblock(256, 16,'att')
        self.pool5_2 = GEblock(512, 16,'att')
        self.pool5_3 = GEblock(1024, 16,'att')
        
        self.conv6 = nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn6 = nn.BatchNorm2d(512)
        
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2, padding=int((2 - 1) // 2))
        
        self.conv7 = nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn7 = nn.BatchNorm2d(512)
        
        self.conv8 = nn.Conv2d(512, 1024, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn8 = nn.BatchNorm2d(1024)
        
        self.pool8 = nn.MaxPool2d(kernel_size=2, stride=2, padding=int((2 - 1) // 2))
        
        self.conv9 = nn.Conv2d(1024, 1024, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn9 = nn.BatchNorm2d(1024)
        
        self.conv1x1_0 = nn.Conv2d(256, 18, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.conv1x1_1 = nn.Conv2d(512, 18, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.conv1x1_2 = nn.Conv2d(1024, 18, kernel_size = 1, stride = 1, padding = 0, bias = False)
        
        self.conv1x1_bn = nn.BatchNorm2d(18)
        self.activate18 = nn.LeakyReLU(0.1, inplace=True)
        
        config = {
        #            gd    gw
        'yolov5s': [0.33, 0.50],
        'yolov5m': [0.67, 0.75],
        'yolov5l': [1.00, 1.00],
        'yolov5x': [1.33, 1.25]
        }

        net_size = config['yolov5l']
        self.YoloV5 = yolov5(nc=1, gd=net_size[0], gw=net_size[1])
        
        self.convyolo1 = nn.Sequential(
                  nn.Conv2d(1024, 256, 1),
                  nn.BatchNorm2d(256),
                  nn.LeakyReLU(0.1, inplace=True)
                )
                
        self.convyolo2 = nn.Sequential(
                  nn.Conv2d(1024, 256, 1),
                  nn.BatchNorm2d(256),
                  nn.LeakyReLU(0.1, inplace=True)
                )
        
        self.convyolo3 = nn.Sequential(
                  nn.Conv2d(256, 512, 3, 1, 1),
                  nn.BatchNorm2d(512),
                  nn.LeakyReLU(0.1, inplace=True)
                )
                
        self.convyolo4 = nn.Sequential(
                  nn.Conv2d(512, 256, 3, 1, 1),
                  nn.BatchNorm2d(256),
                  nn.LeakyReLU(0.1, inplace=True)
                )
                
        self.convyolo5 = nn.Sequential(
                  nn.Conv2d(256, 512, 3, 1, 1),
                  nn.BatchNorm2d(512),
                  nn.LeakyReLU(0.1, inplace=True)
                )
                
        self.convyolo6 = nn.Sequential(
                  nn.Conv2d(256, 128, 1),
                  nn.BatchNorm2d(128),
                  nn.LeakyReLU(0.1, inplace=True)
                )
                
        self.convyolo7 = nn.Sequential(
                  nn.Conv2d(512, 128, 1),
                  nn.BatchNorm2d(128),
                  nn.LeakyReLU(0.1, inplace=True)
                )
                
        self.convyolo8 = nn.Sequential(
                  nn.Conv2d(128, 256, 3, 1, 1),
                  nn.BatchNorm2d(256),
                  nn.LeakyReLU(0.1, inplace=True)
                )
                
        self.Global_extract_layer = Global_extract_layer(256)
        
        self.dynamic_256 = ODConv2d(256, 256, 1, padding=0, groups=256, K=2, r=1/6)
        self.dynamic_512 = ODConv2d(512, 512, 1, padding=0, groups=512, K=2, r=1/6)
        self.dynamic_1024 = ODConv2d(1024, 1024, 1, padding=0, groups=1024, K=2, r=1/6)
        
        self.gcu1 = GraphConv2(batch = 7, h=[64,32,16], w=[64,32,16], d=[256,512,1024], V=[2,4,8,16], outfeatures=[256,512,1024])
        self.Spectral_Discriminator = Spectral_Discriminator(512)

    def meta_forward(self, metax, mask):
        
        spec = self.Spectral_Discriminator(metax)
        spec = nn.Sigmoid()(spec)
        spec = spec.unsqueeze(2)
        spec = spec.unsqueeze(3)
        metax_s = spec * metax
        forgroud = np.zeros((mask.size(0), 3, mask.size(2), mask.size(3)))
        
        for i in range(mask.size(0)):
            out1 = mask[i].squeeze(1)
            toPIL = transforms.ToPILImage()
            out1 = toPIL(out1)
            im_arr_ = toPIL(metax[i])
            y_coords, x_coords = np.nonzero(out1)  
            x_min = x_coords.min()  
            x_max = x_coords.max()  
            y_min = y_coords.min()  
            y_max = y_coords.max()
            canny = np.array(im_arr_)[y_min:y_max,x_min:x_max]
            metax_s_im_arr_ = toPIL(metax_s[i])
            metax_s_im_arr_ = np.asarray(metax_s_im_arr_).copy()
            metax_s_im_arr_[y_min:y_max,x_min:x_max,:] = canny
            forgroud[i] = metax_s_im_arr_.transpose(2, 0, 1)
        forgroud = torch.from_numpy(forgroud).cuda().float()
        metax = forgroud
        metax = self.conv1(metax)
        metax = self.bn1(metax)
        metax = self.activate(metax)
        metax = self.pool1(metax)
        metax = self.conv2(metax)
        metax = self.bn2(metax)
        metax = self.pool2(metax)
        metax = self.conv3(metax)
        metax = self.bn3(metax)
        metax = self.activate(metax)
        metax = self.pool3(metax)
        metax = self.conv4(metax)
        metax = self.bn4(metax)
        metax = self.activate(metax)
        dynamic_weights = self.Global_extract_layer(metax)
        return dynamic_weights

    def detect_forward(self, x, dynamic_weights):
        output = []
        p3, p4, p5, xl, xm, xs = self.YoloV5(x)
        xl_up = xl
        xl_up = self.gcu1(xl_up)
        x_in1 = self.dynamic_1024(xl_up, dynamic_weights[2])
        x_in1 = self.conv1x1_2(x_in1)
        output.append(x_in1.view(x_in1.size(0), x_in1.size(1), x_in1.size(2) * x_in1.size(3)))
        xl_cat_middle = torch.cat((xm, p4), dim = 1) 
        xl_cat_middle = self.convyolo2(xl_cat_middle)
        xl_cat_middle = self.convyolo3(xl_cat_middle)
        xl_cat_middle = self.convyolo4(xl_cat_middle)
        xl_cat_middle = self.convyolo5(xl_cat_middle)
        xl_cat_middle = self.convyolo4(xl_cat_middle)
        xl_up_pmm = self.convyolo5(xl_cat_middle)
        xl_up_pmm = self.gcu1(xl_up_pmm)
        x_in2 = self.dynamic_512(xl_up_pmm, dynamic_weights[1])
        x_in2 = self.conv1x1_1(x_in2)
        output.append(x_in2.view(x_in2.size(0), x_in2.size(1), x_in2.size(2) * x_in2.size(3)))
        xl_up2 = torch.cat((xs, p3), dim = 1)
        xl_up2 = self.convyolo7(xl_up2)
        xl_up2 = self.convyolo8(xl_up2)
        xl_up2 = self.convyolo6(xl_up2)
        xl_up2 = self.convyolo8(xl_up2)
        xl_up2 = self.convyolo6(xl_up2)
        xl_up2 = self.convyolo8(xl_up2)
        xl_up2 = self.gcu1(xl_up2)
        x_in3 = self.dynamic_256(xl_up2, dynamic_weights[0])
        x_in3 = self.conv1x1_0(x_in3)
        output.append(x_in3.view(x_in3.size(0), x_in3.size(1), x_in3.size(2) * x_in3.size(3)))
        return torch.cat(output, 2)

    def forward(self, x, metax, mask, ids=None):
        dynamic_weights = self.meta_forward(metax, mask)
        x = self.detect_forward(x, dynamic_weights)
        return x

    def print_network(self):
        #print_cfg(self.blocks)
        print('---------------------------------------------------------------------')
        print_cfg(self.learnet_blocks)

    def create_network(self, blocks):
        set_params = blocks[0]
        
        init_channel = [int(set_params['channels'])]
        module_list = nn.ModuleList()
        
        routs = []
        ind = -2
        filters = -1
        
        for item in blocks:
            ind +=1
            modules = nn.Sequential()
            
            if item['type'] in ['net']:
                continue
            if item['type'] == 'convolutional':
                bn = int(item['batch_normalize'])
                filters = int(item['filters'])
                size = int(item['size'])
                stride = int(item['stride']) if 'stride' in item else (int(item['stride_y']), int(item['stride_x']))
                pad = (size - 1) // 2 if int(item['pad']) else 0
                dynamic = True if 'dynamic' in item and int(item['dynamic']) == 1 else False
                
                if dynamic:
                    partial = int(item['partial']) if 'partial' in item else None
                    Conv2d = dynamic_conv2d(is_first=True, partial=partial)
                else:
                    Conv2d = nn.Conv2d
            
                modules.add_module('Conv2d', Conv2d(in_channels = init_channel[-1],
                                                        out_channels = filters,
                                                        kernel_size = size,
                                                        stride = stride,
                                                        padding = pad,
                                                        groups=1,
                                                        bias = False))
                if bn:
                    modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))
                if item['activation'] == 'leaky':
                    modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
                elif item['activation'] == 'swish':
                    modules.add_module('', Swish())
            
            elif item['type'] == 'shortcut':
                filters = init_channel[int(item['from'])]
                layer = int(item['from'])
                routs.extend([ind + layer if layer < 0 else layer])
                modules = EmptyModule()
                
            elif item['type'] == 'maxpool':
                size = int(item['size'])
                stride = int(item['stride'])
                maxpool = nn.MaxPool2d(kernel_size=size, stride=stride, padding=int((size - 1) // 2))
                if size == 2 and stride == 1:
                    modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1 , 0, 1)))
                    modules.add_module('MaxPool2d', maxpool)
                else:
                    modules = maxpool
                    
            elif item['type'] == 'upsample':
                modules = nn.Upsample(scale_factor=int(item['stride']), mode='nearest')
                
            elif item['type'] == 'route':
                layers = [int(x) for x in item['layers'].split(',')]
                filters = sum([init_channel[i + 1 if i > 0 else i] for i in layers])
                routs.extend([l if l > 0 else l + ind for l in layers])
                modules = EmptyModule()
            
            elif item['type'] == 'region':
                loss = RegionLossV2()
                anchors = item['anchors'].split(',')
                loss.anchors = [float(i) for i in anchors]
                loss.num_classes = int(item['classes'])
                loss.num_anchors = int(item['num'])
                loss.object_scale = float(item['object_scale'])
                loss.noobject_scale = float(item['noobject_scale'])
                loss.class_scale = float(item['class_scale'])
                loss.coord_scale = float(item['coord_scale'])
                loss.input_size = (self.height, self.width)
                modules = loss
        
            elif item['type'] == 'globalmax':
                modules = GlobalMaxPool2d()
                
            else:
                print('Warning: Unrecognized Layer Type: ' + item['type'])
                
            module_list.append(modules)
            
            init_channel.append(filters)
        return module_list, routs

class Swish(nn.Module):
    def forward(self, x):
        return x.mul_(torch.sigmoid(x))
