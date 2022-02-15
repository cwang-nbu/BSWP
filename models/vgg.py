import torch
import torch.nn as nn
from .stripe import *

__all__ = ['VGG']
default_cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
slope = 0.006


class VGG(nn.Module):
    def __init__(self, num_classes=10, cfg=None):
        super(VGG, self).__init__()
        if cfg is None:
            cfg = default_cfg['VGG16']
        self.features = self._make_layers(cfg)
        self.classifier = Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [FilterStripe(in_channels, x),
                           BatchNorm(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def update_skeleton(self, sr, dic_threshold):
        for key, m in self.named_modules():
            if isinstance(m, FilterStripe):
                out_mask = m.update_skeleton(sr, dic_threshold[key])
            elif isinstance(m, BatchNorm):
                m.update_mask(out_mask)

    def prune(self, dic_threshold):
        in_mask = torch.ones(3) > 0
        for key, m in self.named_modules():
            if isinstance(m, FilterStripe):
                m.prune_in(in_mask)
                in_mask = m.prune_out(dic_threshold[key])
                m._break(dic_threshold[key])
            if isinstance(m, BatchNorm):
                m.prune(in_mask)
            if isinstance(m, Linear):
                m.prune_in(in_mask)


class cal_shape(nn.Module):
    def __init__(self, dic_threshold):
        super(cal_shape, self).__init__()
        self.dic_threshold = dic_threshold

    def forward(self, model, dic_threshold):
        self.dic_threshold = dic_threshold
        loss_stripe_var = 0.
        loss_filter_var = 0.
        loss_mean = 0.
        for key, m in model.named_modules():
            if isinstance(m, FilterStripe) and 'downsample' not in key:
                stripe_var, filter_var, mean = m.cal_skeleton(self.dic_threshold[key])
                loss_stripe_var = loss_stripe_var + stripe_var
                loss_filter_var = loss_filter_var + filter_var
                loss_mean = loss_mean + mean
        return loss_stripe_var, loss_filter_var, loss_mean

class tool():
    def __init__(self,use_function):
        super(tool, self).__init__()
        self.use_function = use_function

    def creat_dic(self, model,threshold):
        dic_threshold= {}
        for key, m in model.named_modules():
            if isinstance(m, FilterStripe) and 'downsample' not in key:
                temp = {key:threshold}
                dic_threshold.update(temp)
        return dic_threshold

    def change_rate(self,model,threshold,dic_threshold):
        dic_threshold_new = {}
        for key, m in model.named_modules():
            if isinstance(m, FilterStripe) and 'downsample' not in key:
                rate_temp = m.cal_skeleton_show(dic_threshold[key])
                rate_temp = rate_temp/9
                if self.use_function == 'convex':
                    value = threshold + slope * rate_temp / (2 - rate_temp)
                    value = value.detach()
                    dic_temp = {key: value}
                    dic_threshold_new.update(dic_temp)
                if self.use_function == 'piecewise':
                    if rate_temp <= 0.2:
                        dic_temp = {key: 0.03}
                        dic_threshold_new.update(dic_temp)
                    elif rate_temp <= 0.25:
                        dic_temp = {key: 0.031}
                        dic_threshold_new.update(dic_temp)
                    elif rate_temp <= 0.3:
                        dic_temp = {key: 0.032}
                        dic_threshold_new.update(dic_temp)
                    elif rate_temp <= 0.35:
                        dic_temp = {key: 0.033}
                        dic_threshold_new.update(dic_temp)
                    elif rate_temp <= 0.4:
                        dic_temp = {key: 0.034}
                        dic_threshold_new.update(dic_temp)
                    elif rate_temp <= 0.45:
                        dic_temp = {key: 0.035}
                        dic_threshold_new.update(dic_temp)
                    else:
                        dic_temp = {key: 0.036}
                        dic_threshold_new.update(dic_temp)

        return dic_threshold_new

    def jianzhilv(self, model, dic_threshold):
        for key, m in model.named_modules():
            if isinstance(m, FilterStripe) and 'downsample' not in key:
                rate_temp = m.cal_skeleton_show(dic_threshold[key])
                rate_temp = rate_temp / 9
                print('{}:{:.4f}'.format(key,rate_temp*100))

