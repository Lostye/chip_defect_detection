# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import cv2


def crop(img_file, mask_file):
    # name, *_ = img_file.split(".")
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_file)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    img_array = np.array(img)
    mask = np.array(mask)
    # img_array = np.array(Image.open(img_file))
    # mask = np.array(Image.open(mask_file))

    # 通过将原图和mask图片归一化值相乘，把背景转成黑色
    # 从mask中随便找一个通道，cat到RGB后面，最后转成RGBA
    # res = np.concatenate((img_array * (mask/255), mask[:, :, [0]]), -1)
    # print(res.shape)
    res = np.concatenate((img_array, mask[:, :, [0]]), -1)
    img = Image.fromarray(res.astype('uint8'), mode='RGBA')
    img.show()
    return img


def label_mask(img_file, mask_file):
    # name, *_ = img_file.split(".")
    img = cv2.imread(img_file)
    mask = cv2.imread(mask_file)
    img = cv2.addWeighted(img, 0.6, mask, 0.4, 0)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img, contours, -1, (25, 255, 255), 5)
    # 打开画了轮廓之后的图像
    # cv2.imshow('mask', img)
    # cv2.waitKey(0)

    # cv2.destroyAllWindows()
    cv2.imwrite('./static/example.png', img)
    return img


if __name__ == "__main__":
    import os

    # model = "u2net"
    model = "u2netp"

    img_root = "test_data/test_images"
    mask_root = "test_data/{}_results".format(model)
    crop_root = "test_data/{}_crops".format(model)
    os.makedirs(crop_root, mode=0o775, exist_ok=True)

    for img_file in os.listdir(img_root):
        print("crop image {}".format(img_file))
        name, *_ = img_file.split(".")
        res = label_mask(
            os.path.join(img_root, img_file),
            os.path.join(mask_root, name + ".png")
        )
        cv2.imwrite(os.path.join(crop_root, name + "_crop.png"), res)
        # res.save(os.path.join(crop_root, name + "_crop.png"))
        # exit()
