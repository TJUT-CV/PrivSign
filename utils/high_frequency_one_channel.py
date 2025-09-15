import torch
from torchjpeg import dct
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


""" FCS on 64 channel """
def images_to_batch(x):

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

    ## Y in YCbCr
    dct_block_y_dc = dct_block[:, :1, 0:1, :, :]
    finde_tensor_y = dct_block_y_dc.squeeze().numpy()
    # plt.imsave('.//visual/64_one_channel_for_00/00_y_DC.png', finde_tensor_y, cmap='viridis')
    plt.imsave('.//visual/DC_one_channels_00_26/00_y_DC.png', finde_tensor_y, cmap='viridis')

    # dct_block_y_1 = dct_block[:, :1, 14:15, :, :]
    # finde_tensor = dct_block_y_1.squeeze().numpy()
    # plt.imsave('.//visual/64_one_channel_for_26/26_y_H14.png', finde_tensor, cmap='viridis')
    #
    # dct_block_y_4 = dct_block[:, :1, 28:29, :, :]
    # finde_tensor = dct_block_y_4.squeeze().numpy()
    # plt.imsave('.//visual/64_one_channel_for_26/26_y_H27.png', finde_tensor, cmap='viridis')
    #
    # dct_block_y_5 = dct_block[:, :1, 44:45, :, :]
    # finde_tensor = dct_block_y_5.squeeze().numpy()
    # plt.imsave('.//visual/64_one_channel_for_26/26_y_H43.png', finde_tensor, cmap='viridis')
    #
    # dct_block_y_7 = dct_block[:, :1, 59:60, :, :]
    # finde_tensor = dct_block_y_7.squeeze().numpy()
    # plt.imsave('.//visual/64_one_channel_for_26/26_y_H59.png', finde_tensor, cmap='viridis')


    ## Cb in YCbCr
    dct_block_y_cb = dct_block[:, 1:2, 0:1, :, :]
    finde_tensor_cb = dct_block_y_cb.squeeze().numpy()
    # plt.imsave('.//visual/64_one_channel_for_00/00_cb_DC.png', finde_tensor_cb, cmap='viridis')
    plt.imsave('.//visual/DC_one_channels_00_26/00_cb_DC.png', finde_tensor_cb, cmap='viridis')

    # dct_block_cb_1 = dct_block[:, 1:2, 28:29, :, :]
    # finde_tensor = dct_block_cb_1.squeeze().numpy()
    # plt.imsave('.//visual/64_one_channel_for_26/26_cb_H28.png', finde_tensor, cmap='viridis')
    #
    # dct_block_cb_3 = dct_block[:, 1:2, 44:45, :, :]
    # finde_tensor = dct_block_cb_3.squeeze().numpy()
    # plt.imsave('.//visual/64_one_channel_for_26/26_cb_H44.png', finde_tensor, cmap='viridis')
    #
    # dct_block_cb_4 = dct_block[:, 1:2, 52:53, :, :]
    # finde_tensor = dct_block_cb_4.squeeze().numpy()
    # plt.imsave('.//visual/64_one_channel_for_26/26_cb_H52.png', finde_tensor, cmap='viridis')


    ## Cr in YCbCr
    dct_block_y_cr = dct_block[:, 2:3, 0:1, :, :]
    finde_tensor_cr = dct_block_y_cr.squeeze().numpy()
    # plt.imsave('.//visual/64_one_channel_for_00/00_cr_DC.png', finde_tensor_cr, cmap='viridis')
    plt.imsave('.//visual/DC_one_channels_00_26/00_cr_DC.png', finde_tensor_cr, cmap='viridis')

    # dct_block_cr_1 = dct_block[:, 2:3, 28:29, :, :]
    # finde_tensor = dct_block_cr_1.squeeze().numpy()
    # plt.imsave('.//visual/64_one_channel_for_26/26_cr_H28.png', finde_tensor, cmap='viridis')
    #
    # dct_block_cr_3 = dct_block[:, 2:3, 44:45, :, :]
    # finde_tensor = dct_block_cr_3.squeeze().numpy()
    # plt.imsave('.//visual/64_one_channel_for_26/26_cr_H44.png', finde_tensor, cmap='viridis')
    #
    # dct_block_cr_4 = dct_block[:, 2:3, 52:53, :, :]
    # finde_tensor = dct_block_cr_4.squeeze().numpy()
    # plt.imsave('.//visual/64_one_channel_for_26/26_cr_H52.png', finde_tensor, cmap='viridis')

###
def read_and_preprocess_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  #
        transforms.ToTensor(),  #
    ])
    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0)  #
    return image_tensor


""" back up"""
def main(image_path, idx):
    image_tensor = read_and_preprocess_image(image_path)
    channel_to_save = 48
    images_to_batch(image_tensor)



if __name__ == "__main__":
    # for i in [2, 5, 10, 15, 17, 20, 25, 30]:
    # main('/hdd1/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-256x256px/train/01April_2010_Thursday_heute_default-0/1/01April_2010_Thursday_heute.avi_pid0_fn000026-0.png', 0)
    main('/hdd1/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-256x256px/train/01April_2010_Thursday_heute_default-0/1/01April_2010_Thursday_heute.avi_pid0_fn000000-0.png', 0)



