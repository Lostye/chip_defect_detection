import os
from PIL import Image

file_dir = "F:\BaiduNetdiskDownload\dataset1"
for i in os.listdir(file_dir):
    dir_path = os.path.join(file_dir,i)
    if os.path.isdir(dir_path):

        file_label = Image.open(os.path.join(dir_path,"label.png"))
        file_name = i.split("_")[0]
        if not os.path.exists("label"):
            os.makedirs("label")
        file_label.save("label/{}.png".format(file_name))
        print(file_name)