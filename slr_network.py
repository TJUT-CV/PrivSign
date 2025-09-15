import pdb
import copy
import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.criterions import SeqKD
from modules import BiLSTMLayer, TemporalConv
# from modules import BiLSTMLayer, TemporalConv_tlp
# import modules.SEN_resnet as resnet
# import modules.TLP_VAC_SMKD_resnet as resnet
import modules.CorrNet_resnet as resnet
from torchjpeg import dct
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class SLRModel(nn.Module):
    def __init__(
            self, num_classes, c2d_type, conv_type, use_bn=False,
            hidden_size=1024, gloss_dict=None, loss_weights=None,
            weight_norm=True, share_classifier=True
    ):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        # self.conv2d = getattr(models, c2d_type)(pretrained=True)
        self.conv2d = getattr(resnet, c2d_type)()
        # selected_layers = list(self.conv2d.children())[4:]
        # custom_resnet = nn.Sequential(*selected_layers)
        # self.conv2d = custom_resnet
        # self.conv2d[5] = Identity()
        self.conv2d.fc = Identity()
        """ origin"""
        self.conv1d = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        """ tlp """
        # self.conv1d = TemporalConv_tlp(input_size=512,
        #                                hidden_size=hidden_size,
        #                                conv_type=conv_type,
        #                                 use_bn=use_bn,
        #                                num_classes=num_classes)
        self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)
        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier
        #self.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])


        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        # x = images_to_batch(x)
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, x, len_x, label=None, label_lgt=None):
        def images_to_batch(x):
            # start_time = time.time()
            # x = x.to(torch.float32)
            x = (x + 1) / 2 * 255

            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            if x.shape[1] != 3:
                print("Wrong input, Channel should equals to 3")
                return
            x = dct.to_ycbcr(x)  # comvert RGB to YCBCR
            x -= 128
            bs, ch, h, w = x.shape
            block_num = h // 8
            x = x.view(bs * ch, 1, h, w)
            x = F.unfold(x, kernel_size=(8, 8), dilation=1, padding=0,
                         stride=(8, 8))
            x = x.transpose(1, 2)
            x = x.view(bs, ch, -1, 8, 8)
            dct_block = dct.block_dct(x)
            dct_block = dct_block.view(bs, ch, block_num, block_num, 64).permute(0, 1, 4, 2, 3)

            """ FCS """
            dct_block_y_1 = dct_block[:, :1, 3:8, :, :]
            dct_block_y_2 = dct_block[:, :1, 10:15, :, :]
            dct_block_y_3 = dct_block[:, :1, 17:22, :, :]
            dct_block_y_4 = dct_block[:, :1, 24:29, :, :]
            dct_block_y_5 = dct_block[:, :1, 32:36, :, :]
            dct_block_y_6 = dct_block[:, :1, 40:43, :, :]
            dct_block_y_7 = dct_block[:, :1, 48:50, :, :]
            dct_block_y_8 = dct_block[:, :1, 56:57, :, :]
            dct_block_y = torch.cat((dct_block_y_1, dct_block_y_2, dct_block_y_3, dct_block_y_4, dct_block_y_5,
                                     dct_block_y_6, dct_block_y_7, dct_block_y_8), dim=2)
            dct_block_cb_1 = dct_block[:, 1:2, 3:7, :, :]
            dct_block_cb_2 = dct_block[:, 1:2, 10:13, :, :]
            dct_block_cb_3 = dct_block[:, 1:2, 17:20, :, :]
            dct_block_cb_4 = dct_block[:, 1:2, 24:27, :, :]
            dct_block_cb_5 = dct_block[:, 1:2, 32:34, :, :]
            dct_block_cb_6 = dct_block[:, 1:2, 40:41, :, :]
            dct_block_cb_7 = dct_block[:, 1:2, 48:49, :, :]
            dct_block_cb = torch.cat((dct_block_cb_1, dct_block_cb_2, dct_block_cb_3, dct_block_cb_4, dct_block_cb_5,
                                      dct_block_cb_6, dct_block_cb_7), dim=2)
            dct_block_cr_1 = dct_block[:, 2:, 3:7, :, :]
            dct_block_cr_2 = dct_block[:, 2:, 10:13, :, :]
            dct_block_cr_3 = dct_block[:, 2:, 17:20, :, :]
            dct_block_cr_4 = dct_block[:, 2:, 24:27, :, :]
            dct_block_cr_5 = dct_block[:, 2:, 32:34, :, :]
            dct_block_cr_6 = dct_block[:, 2:, 40:41, :, :]
            dct_block_cr_7 = dct_block[:, 2:, 48:49, :, :]
            dct_block_cr = torch.cat((dct_block_cr_1, dct_block_cr_2, dct_block_cr_3, dct_block_cr_4, dct_block_cr_5,
                                      dct_block_cr_6, dct_block_cr_7), dim=2)
            """ """
            dct_block = torch.cat((dct_block_y, dct_block_cb, dct_block_cr), dim=2)  # remove DC
            dct_block = dct_block.reshape(bs, -1, block_num, block_num)

            return dct_block
        """ CorrNet """
        if len(x.shape) == 5:
            # videos
            batch, temp, channel, height, width = x.shape
            x = x.reshape(batch*temp, channel, height, width)
            x = images_to_batch(x)
            x = x.reshape(batch, -1, 64, 56, 56)
            batch, temp, channel, height, width = x.shape
            #inputs = x.reshape(batch * temp, channel, height, width)
            #framewise = self.masked_bn(inputs, len_x)
            #framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)
            framewise = self.conv2d(x.permute(0,2,1,3,4)).view(batch, temp, -1).permute(0,2,1) # btc -> bct


        # """ SEN """
        # if len(x.shape) == 5:
        #     # videos
        #     batch, temp, channel, height, width = x.shape
        #     #inputs = x.reshape(batch * temp, channel, height, width)
        #     #framewise = self.masked_bn(inputs, len_x)
        #     framewise = self.conv2d(x.permute(0,2,1,3,4))
        #     framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)

        # """ TLP """
        # if len(x.shape) == 5:
        #     # videos
        #     batch, temp, channel, height, width = x.shape
        #     inputs = x.reshape(batch * temp, channel, height, width)
        #     framewise = self.masked_bn(inputs, len_x)
        #     framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)

        else:
            # frame-wise features
            framewise = x

        conv1d_outputs = self.conv1d(framewise, len_x)
        # x: T, B, C
        x = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']
        tm_outputs = self.temporal_model(x, lgt)
        outputs = self.classifier(tm_outputs['predictions'])
        pred = None if self.training \
            else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
        conv_pred = None if self.training \
            else self.decoder.decode(conv1d_outputs['conv_logits'], lgt, batch_first=False, probs=False)

        return {
            #"framewise_features": framewise,
            #"visual_features": x,
            "feat_len": lgt,
            "conv_logits": conv1d_outputs['conv_logits'],
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
        }

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0
        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'SeqCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'Dist':
                loss += weight * self.loss['distillation'](ret_dict["conv_logits"],
                                                           ret_dict["sequence_logits"].detach(),
                                                           use_blank=False)
        return loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        return self.loss
