import os
import lmdb
from PIL import Image
from xml.dom import minidom
from io import BytesIO

def checkImageIsValid(file):
    valid = True
    try:
        Image.open(file).load()
    except OSError:
        valid = False
    return valid

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)

def is_alpha_numeric(label, alphabet):
    for i in label:
        if not i in alphabet:
            return False
    return True

def main(outputPath, checkValid=True):

    root_path = "/root/IC03_SceneTrialTest/"  # 这里改是 test 还是 train
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1

    doc = minidom.parse(root_path + "words.xml")
    img_tags = doc.getElementsByTagName("image")
    for img_tag in img_tags:
        img_name = img_tag.getElementsByTagName("imageName")[0].firstChild.data
        # print(img_name)

        path_key = img_name.split('/')[1].split('.')[0]

        img = Image.open(root_path + img_name)
        # img.save("./001.jpg")

        rect_tags = img_tag.getElementsByTagName("taggedRectangle")
        for rect_tag in rect_tags:
            label = rect_tag.getElementsByTagName('tag')[0].firstChild.data
            if not is_alpha_numeric(label, alphabet) or len(label) < 3:
                # print("ignore ", label)
                continue
            x = float(rect_tag.getAttribute("x"))
            y = float(rect_tag.getAttribute("y"))
            width = float(rect_tag.getAttribute("width"))
            height = float(rect_tag.getAttribute("height"))

            b_img = BytesIO()
            img.crop((x, y, x+width, y+height)).save(b_img, format='jpeg')

            imageKey = "image_{:09d}".format(cnt)
            labelKey = "label_{:09d}".format(cnt)
            pathKey = "path_{:09d}".format(cnt)
            cache[imageKey] = b_img.getvalue()
            cache[labelKey] = (label).encode(encoding="utf-8")
            cache[pathKey] = (path_key).encode(encoding="utf-8")
            if cnt % 1000 == 0:
                writeCache(env, cache)
                cache = {}
                print('Written %d' % (cnt))
            cnt += 1
    
    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created %s with %d samples' % (outputPath, nSamples))


if __name__ == '__main__':
    main("/root/project/data/lmdb_ic03_SceneTest")  # 这里改是 test 还是 train
