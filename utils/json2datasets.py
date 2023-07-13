import os
import glob

path = r"F:\BaiduNetdiskDownload\dataset1"
json_file = glob.glob(os.path.join(path,"*.json"))
print(json_file)
for file in json_file:
    os.system("labelme_json_to_dataset.exe %s"%(file))