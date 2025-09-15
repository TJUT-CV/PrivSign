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
    dct_block_y_1 = dct_block[:, :1, 14:16, :, :]
    dct_block_y_2 = dct_block[:, :1, 19:24, :, :]
    dct_block_y_3 = dct_block[:, :1, 27:32, :, :]
    dct_block_y_4 = dct_block[:, :1, 35:40, :, :]
    dct_block_y_5 = dct_block[:, :1, 43:48, :, :]
    dct_block_y_6 = dct_block[:, :1, 51:56, :, :]
    dct_block_y_7 = dct_block[:, :1, 59:64, :, :]

    dct_block_y = torch.cat((dct_block_y_1, dct_block_y_2, dct_block_y_3, dct_block_y_4, dct_block_y_5,
                             dct_block_y_6, dct_block_y_7), dim=2)
    ## Cb in YCbCr
    dct_block_cb_1 = dct_block[:, 1:2, 28:32, :, :]
    dct_block_cb_2 = dct_block[:, 1:2, 36:40, :, :]
    dct_block_cb_3 = dct_block[:, 1:2, 44:48, :, :]
    dct_block_cb_4 = dct_block[:, 1:2, 52:56, :, :]
    dct_block_cb = torch.cat((dct_block_cb_1, dct_block_cb_2, dct_block_cb_3, dct_block_cb_4), dim=2)

    ## Cr in YCbCr
    dct_block_cr_1 = dct_block[:, 2:3, 28:32, :, :]
    dct_block_cr_2 = dct_block[:, 2:3, 36:40, :, :]
    dct_block_cr_3 = dct_block[:, 2:3, 44:48, :, :]
    dct_block_cr_4 = dct_block[:, 2:3, 52:56, :, :]

    dct_block_cr = torch.cat((dct_block_cr_1, dct_block_cr_2, dct_block_cr_3, dct_block_cr_4), dim=2)


    dct_block_1 = torch.cat((dct_block_y, dct_block_cb, dct_block_cr), dim=2)  # remove DC
    dct_block_1 = dct_block_1.reshape(bs, -1, block_num, block_num)


    return dct_block_1

""" FCS on 128 channel """
# def images_to_batch(x):
#     # start_time = time.time()
#     # x = x.to(torch.float32)
#     x = (x + 1) / 2 * 255
#
#     x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
#     if x.shape[1] != 3:
#         print("Wrong input, Channel should equals to 3")
#         return
#     x = dct.to_ycbcr(x)  # comvert RGB to YCBCR
#     x -= 128
#     bs, ch, h, w = x.shape
#     block_num = h // 8
#     x = x.view(bs * ch, 1, h, w)
#     x = F.unfold(x, kernel_size=(8, 8), dilation=1, padding=0,
#                  stride=(8, 8))
#     x = x.transpose(1, 2)
#     x = x.view(bs, ch, -1, 8, 8)
#     dct_block = dct.block_dct(x)
#     dct_block = dct_block.view(bs, ch, block_num, block_num, 64).permute(0, 1, 4, 2, 3)
#
#     dct_block_y_1 = dct_block[:, :1, 4:8, :, :] #4
#     dct_block_y_2 = dct_block[:, :1, 11:16, :, :]  #
#     dct_block_y_3 = dct_block[:, :1, 18:24, :, :]  #
#     dct_block_y_4 = dct_block[:, :1, 25:32, :, :]  #
#     dct_block_y_5 = dct_block[:, :1, 32:39, :, :]  #
#     dct_block_y_6 = dct_block[:, :1, 40:45, :, :]  #
#     dct_block_y_7 = dct_block[:, :1, 48:52, :, :]  #
#     dct_block_y_8 = dct_block[:, :1, 56:60, :, :]  #
#     dct_block_y = torch.cat((dct_block_y_1, dct_block_y_2, dct_block_y_3, dct_block_y_4, dct_block_y_5,
#                              dct_block_y_6, dct_block_y_7, dct_block_y_8), dim=2)
#
#     dct_block_cb_1 = dct_block[:, 1:2, 6:8, :, :]
#     dct_block_cb_2 = dct_block[:, 1:2, 11:16, :, :]
#     dct_block_cb_3 = dct_block[:, 1:2, 19:24, :, :]
#     dct_block_cb_4 = dct_block[:, 1:2, 27:32, :, :]
#     dct_block_cb_5 = dct_block[:, 1:2, 35:40, :, :]
#     dct_block_cb_6 = dct_block[:, 1:2, 43:48, :, :]
#     dct_block_cb_7 = dct_block[:, 1:2, 48:56, :, :]
#     dct_block_cb_8 = dct_block[:, 1:2, 56:64, :, :]
#     dct_block_cb = torch.cat((dct_block_cb_1, dct_block_cb_2, dct_block_cb_3, dct_block_cb_4, dct_block_cb_5,
#                               dct_block_cb_6, dct_block_cb_7, dct_block_cb_8), dim=2)
#
#     dct_block_cr_1 = dct_block[:, 2:3, 6:8, :, :]
#     dct_block_cr_2 = dct_block[:, 2:3, 11:16, :, :]
#     dct_block_cr_3 = dct_block[:, 2:3, 19:24, :, :]
#     dct_block_cr_4 = dct_block[:, 2:3, 27:32, :, :]
#     dct_block_cr_5 = dct_block[:, 2:3, 35:40, :, :]
#     dct_block_cr_6 = dct_block[:, 2:3, 43:48, :, :]
#     dct_block_cr_7 = dct_block[:, 2:3, 48:56, :, :]
#     dct_block_cr_8 = dct_block[:, 2:3, 56:64, :, :]
#     dct_block_cr = torch.cat((dct_block_cr_1, dct_block_cr_2, dct_block_cr_3, dct_block_cr_4, dct_block_cr_5,
#                               dct_block_cr_6, dct_block_cr_7, dct_block_cr_8), dim=2)
#     dct_block = torch.cat((dct_block_y, dct_block_cb, dct_block_cr), dim=2)  # remove DC
#     dct_block = dct_block.reshape(bs, -1, block_num, block_num)
#     return dct_block

""" FCS on 160 channel """

# def images_to_batch(x):
#
#     x = (x + 1) / 2 * 255
#
#     x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
#     if x.shape[1] != 3:
#         print("Wrong input, Channel should equals to 3")
#         return
#     x = dct.to_ycbcr(x)  # comvert RGB to YCBCR
#     x -= 128
#     bs, ch, h, w = x.shape
#     block_num = h // 8
#     x = x.view(bs * ch, 1, h, w)
#     x = F.unfold(x, kernel_size=(8, 8), dilation=1, padding=0,
#                  stride=(8, 8))
#     x = x.transpose(1, 2)
#     x = x.view(bs, ch, -1, 8, 8)
#     dct_block = dct.block_dct(x)
#     dct_block = dct_block.view(bs, ch, block_num, block_num, 64).permute(0, 1, 4, 2, 3)
#     dct_block_y_1 = dct_block[:, :1, 4:8, :, :]  #
#     dct_block_y_2 = dct_block[:, :1, 11:16, :, :]
#     dct_block_y_3 = dct_block[:, :1, 18:24, :, :]
#     dct_block_y_4 = dct_block[:, :1, 25:64, :, :]
#     dct_block_y = torch.cat((dct_block_y_1, dct_block_y_2, dct_block_y_3, dct_block_y_4), dim=2)
#
#     dct_block_cb_1 = dct_block[:, 1:2, 4:8, :, :]
#     dct_block_cb_2 = dct_block[:, 1:2, 11:16, :, :]
#     dct_block_cb_3 = dct_block[:, 1:2, 18:24, :, :]
#     dct_block_cb_4 = dct_block[:, 1:2, 25:63, :, :]
#     dct_block_cb = torch.cat((dct_block_cb_1, dct_block_cb_2, dct_block_cb_3, dct_block_cb_4), dim=2)
#
#     dct_block_cr_1 = dct_block[:, 2:3, 4:8, :, :]
#     dct_block_cr_2 = dct_block[:, 2:3, 11:16, :, :]
#     dct_block_cr_3 = dct_block[:, 2:3, 18:24, :, :]
#     dct_block_cr_4 = dct_block[:, 2:3, 25:63, :, :]
#     dct_block_cr = torch.cat((dct_block_cr_1, dct_block_cr_2, dct_block_cr_3, dct_block_cr_4), dim=2)
#     dct_block_1 = torch.cat((dct_block_y, dct_block_cb, dct_block_cr), dim=2)  # remove DC
#     dct_block_1 = dct_block_1.reshape(bs, -1, block_num, block_num)
#
#
#     return dct_block_1
# """ back up"""
# def images_to_batch(x):
#     # start_time = time.time()
#     # x = x.to(torch.float32)
#     x = (x + 1) / 2 * 255
#
#     x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
#     if x.shape[1] != 3:
#         print("Wrong input, Channel should equals to 3")
#         return
#     x = dct.to_ycbcr(x)
#     x= dct.to_rgb(x)# comvert RGB to YCBCR
#     x -= 128
#     bs, ch, h, w = x.shape
#     block_num = h // 8
#     x = x.view(bs * ch, 1, h, w)
#     x = F.unfold(x, kernel_size=(8, 8), dilation=1, padding=0,
#                  stride=(8, 8))
#     x = x.transpose(1, 2)
#     x = x.view(bs, ch, -1, 8, 8)
#     dct_block = dct.block_dct(x)
#     dct_block = dct_block.view(bs, ch, block_num, block_num, 64).permute(0, 1, 4, 2, 3)
#     dct_block_y_1 = dct_block[:, :1, 3:8, :, :]
#     dct_block_y_2 = dct_block[:, :1, 10:15, :, :]
#     dct_block_y_3 = dct_block[:, :1, 17:22, :, :]
#     dct_block_y_4 = dct_block[:, :1, 24:29, :, :]
#     dct_block_y_5 = dct_block[:, :1, 32:36, :, :]
#     dct_block_y_6 = dct_block[:, :1, 40:43, :, :]
#     dct_block_y_7 = dct_block[:, :1, 48:50, :, :]
#     dct_block_y_8 = dct_block[:, :1, 56:57, :, :]
#     dct_block_y = torch.cat((dct_block_y_1, dct_block_y_2, dct_block_y_3, dct_block_y_4, dct_block_y_5,
#                              dct_block_y_6, dct_block_y_7, dct_block_y_8), dim=2)
#     # dct_block_y = torch.cat((dct_block_y_1, dct_block_y_2), dim=2)
#     dct_block_cb_1 = dct_block[:, 1:2, 3:7, :, :]
#     dct_block_cb_2 = dct_block[:, 1:2, 10:13, :, :]
#     dct_block_cb_3 = dct_block[:, 1:2, 17:20, :, :]
#     dct_block_cb_4 = dct_block[:, 1:2, 24:27, :, :]
#     dct_block_cb_5 = dct_block[:, 1:2, 32:34, :, :]
#     dct_block_cb_6 = dct_block[:, 1:2, 40:41, :, :]
#     dct_block_cb_7 = dct_block[:, 1:2, 48:49, :, :]
#     dct_block_cb = torch.cat((dct_block_cb_1, dct_block_cb_2, dct_block_cb_3, dct_block_cb_4, dct_block_cb_5,
#                               dct_block_cb_6, dct_block_cb_7), dim=2)
#     # dct_block_cb = torch.cat((dct_block_cb_1, dct_block_cb_2), dim=2)
#     dct_block_cr_1 = dct_block[:, 2:, 3:7, :, :]
#     dct_block_cr_2 = dct_block[:, 2:, 10:13, :, :]
#     dct_block_cr_3 = dct_block[:, 2:, 17:20, :, :]
#     dct_block_cr_4 = dct_block[:, 2:, 24:27, :, :]
#     dct_block_cr_5 = dct_block[:, 2:, 32:34, :, :]
#     dct_block_cr_6 = dct_block[:, 2:, 40:41, :, :]
#     dct_block_cr_7 = dct_block[:, 2:, 48:49, :, :]
#     dct_block_cr = torch.cat((dct_block_cr_1, dct_block_cr_2, dct_block_cr_3, dct_block_cr_4, dct_block_cr_5,
#                               dct_block_cr_6, dct_block_cr_7), dim=2)
#
#     dct_block = torch.cat((dct_block_y, dct_block_cb, dct_block_cr), dim=2)  # remove DC
#     dct_block = dct_block.reshape(bs, -1, block_num, block_num)
#     # dct_block = dct_block.reshape(2, 64, -1, block_num, block_num)
#
#     return dct_block

### 显示信道
def read_and_preprocess_image(image_path):
    # 使用Pillow库加载图片
    image = Image.open(image_path)

    # 定义转换流程
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # 将图片大小调整为224x224
        transforms.ToTensor(),  # 将图片转换为Tensor
    ])

    # 应用转换
    image_tensor = preprocess(image)
    # 在批次维度上添加一个维度，以便与之前的函数兼容
    image_tensor = image_tensor.unsqueeze(0)  # 增加批次维度
    return image_tensor


""" back up"""
# def main(image_path):
#     # 
#     image_tensor = read_and_preprocess_image(image_path)
#     channel_to_save = 48
#     # 
#     processed_batch = images_to_batch(image_tensor)
#     processed_batch = processed_batch.squeeze()
#     # finde_tensor = torch.mean(processed_batch, dim=0)
#     finde_tensor = F.interpolate(processed_batch.unsqueeze(0), (224, 224), mode='bilinear', align_corners=False)
#     finde_tensor = finde_tensor.squeeze().numpy()
#     plt.imshow(finde_tensor[channel_to_save], cmap='gray')  # 使用Viridis色彩映射
#     # plt.title(f'Channel {channel_to_save}')
#     plt.axis('off')
#     plt.savefig('.//TestProject/channel_1_48_gray.png', bbox_inches='tight', pad_inches=0)
def main(image_path, idx):
    # 读取和预处理图像
    image_tensor = read_and_preprocess_image(image_path)
    channel_to_save = 48
    # 调用修改后的 images_to_batch 函数处理图像张量
    processed_batch = images_to_batch(image_tensor)
    # processed_batch = processed_batch.squeeze()
    # finde_tensor = torch.mean(processed_batch, dim=0)
    # finde_tensor = F.interpolate(processed_batch.unsqueeze(0), (224, 224), mode='bilinear', align_corners=False)
    finde_tensor = processed_batch.squeeze().numpy()
    finde_tensor = np.mean(finde_tensor, axis=0)

    # plt.imshow(finde_tensor, cmap='gray')
    # plt.axis('off')
    # plt.imsave(f'./visual/64channel/{idx}_H.png', finde_tensor, cmap='viridis')
    # plt.imsave(f'./visual/128channel/{idx}_H.png', finde_tensor, cmap='viridis')
    # plt.imsave(f'./visual/160channel/{idx}_H.png', finde_tensor, cmap='viridis')
    plt.imsave('./visual/64channel/00_H.png', finde_tensor, cmap='viridis')
    # plt.imsave('./visual/01April_2010_Thursday_heute.avi_pid0_fn000026-0_H.png', finde_tensor, cmap='viridis')

    # plt.savefig('.//TestProject/gray_1.png', bbox_inches='tight', pad_inches=0)
    # finde_tensor = finde_tensor.squeeze()
    # plt.imshow(finde_tensor.numpy(), cmap='viridis')
    # plt.axis('off')
    # plt.savefig('.//visual/9_1.png', bbox_inches='tight', pad_inches=0)
    # plt.show()

    # 这里可以添加更多的处理或者输出逻辑
    # print("处理后的批次数据维度：", processed_batch.shape)

if __name__ == "__main__":
    # for i in [2, 5, 10, 15, 17, 20, 25, 30]:
        # main(f'./-visual/attack_pic/Origin_pic/{i}.png', i)
    # main('/hdd1/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-256x256px/train/01April_2010_Thursday_heute_default-0/1/01April_2010_Thursday_heute.avi_pid0_fn000026-0.png', 0)
    main('/hdd1/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-256x256px/train/01April_2010_Thursday_heute_default-0/1/01April_2010_Thursday_heute.avi_pid0_fn000000-0.png', 0)



