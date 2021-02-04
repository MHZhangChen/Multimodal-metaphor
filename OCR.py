from aip import AipOcr
import os, csv

""" 你的 APPID AK SK """
APP_ID = '21288604'
API_KEY = 'CyMtQTBU6Dx7QmC04mE2HaiU'
SECRET_KEY = 'KWWcVwOkhxl0OqZLMAdlGEGrtdhuGDHq'

client = AipOcr(APP_ID, API_KEY, SECRET_KEY)

options = {}
options["language_type"] = "ENG"
# options["detect_direction"] = "true"
# options["detect_language"] = "true"
# options["probability"] = "true"

""" 读取图片 """
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

write = [['id', 'text']]

root = "FB_pic"
flist = os.listdir(root)

for i in range(len(flist)):
    id = "FB" + flist[i].split('.')[0]

    path = os.path.join(root, flist[i])
    image = get_file_content(path)

    try:
        # print(client.basicGeneral(image, options))
        result = client.basicGeneral(image, options)['words_result']
        text = ""
        for res in result:
            text += res['words'] + ' '

        print(i, id, text.lower())
        write.append([id, text.lower()])
    except KeyError:
        print(id, client.basicGeneral(image, options))
        pass
    except OSError:
        print("OSError")
        pass

print(len(write) - 1)

with open('FB_ocr.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    for line in write:
        writer.writerow(line)