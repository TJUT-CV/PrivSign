import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path ='/hdd1/phoenix2014-release/phoenix-2014-multisigner/features/fullFrame-256x256px/train/01April_2010_Thursday_heute_default-0/1/01April_2010_Thursday_heute.avi_pid0_fn000000-0.png'

image_bgr = cv2.imread(image_path)  #
image_ycbcr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
#
Y, Cb, Cr = cv2.split(image_ycbcr)
Cb_normalized = cv2.normalize(Cb, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
Cr_normalized = cv2.normalize(Cr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
#
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
# axes[0].set_title('Original Image')
axes[0].axis('off')

#
axes[1].imshow(Y, cmap='gray')
# axes[1].set_title('Y Channel (Luminance)')
axes[1].axis('off')
extent = axes[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
plt.savefig(f'.//visual/YCbCr/Y00.png', bbox_inches=extent)
#
axes[2].imshow(Cb_normalized, cmap='jet')#jet
# axes[2].set_title('Cb Channel (Chroma Blue - Pseudocolor)')
axes[2].axis('off')
extent = axes[2].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
plt.savefig(f'.//visual/YCbCr/Cb00.png', bbox_inches=extent)
#
axes[3].imshow(Cr_normalized, cmap='jet')#jet
# axes[3].set_title('Cr Channel (Chroma Red - Pseudocolor)')
axes[3].axis('off')
extent = axes[3].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
plt.savefig(f'.//visual/YCbCr/Cr00.png', bbox_inches=extent)

plt.tight_layout()
plt.show()

print(f"Converted image saved at .//visual/YCbCr")

