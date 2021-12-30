import os
import lmdb  # install lmdb by "pip install lmdb"
# import cv2
import numpy as np

from PIL import Image


def checkImageIsValid(file):
    valid = True
    try:
        Image.open(file).load()
    except OSError:
        valid = False
    return valid


# def checkImageIsValid(imageBin):
#     if imageBin is None:
#         return False
#     imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
#     img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
#     imgH, imgW = img.shape[0], img.shape[1]
#     if imgH * imgW == 0:
#         return False
#     return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)


# def createDataset(outputPath,
#                   imagePathList,
#                   labelList,
#                   lexiconList=None,
#                   checkValid=True):
#     """
#     Create LMDB dataset for CRNN training.
#     ARGS:
#         outputPath    : LMDB output path
#         imagePathList : list of image path
#         labelList     : list of corresponding groundtruth texts
#         lexiconList   : (optional) list of lexicon lists
#         checkValid    : if true, check the validity of every image
#     """
#     assert (len(imagePathList) == len(labelList))
#     nSamples = len(imagePathList)
#     env = lmdb.open(outputPath, map_size=1099511627776)
#     cache = {}
#     cnt = 1
#     for i in range(nSamples):
#         imagePath = imagePathList[i]
#         label = labelList[i]
#         if not os.path.exists(imagePath):
#             print('%s does not exist' % imagePath)
#             continue
#         with open(imagePath, 'r') as f:
#             imageBin = f.read()
#         if checkValid:
#             if not checkImageIsValid(imageBin):
#                 print('%s is not a valid image' % imagePath)
#                 continue

#         imageKey = "image_{:09d}".format(cnt)
#         labelKey = "label_{:09d}".format(cnt)
#         cache[imageKey] = imageBin
#         cache[labelKey] = label
#         if lexiconList:
#             lexiconKey = 'lexicon-%09d' % cnt
#             cache[lexiconKey] = ' '.join(lexiconList[i])
#         if cnt % 1000 == 0:
#             writeCache(env, cache)
#             cache = {}
#             print('Written %d / %d' % (cnt, nSamples))
#         cnt += 1
#     nSamples = cnt - 1
#     cache['num-samples'] = str(nSamples)
#     writeCache(env, cache)
#     print('Created dataset with %d samples' % nSamples)


def main(outputPath, checkValid=True):

    root_path = "/root/synth/mnt/ramdisk/max/90kDICT32px/"

    file = open("/root/synth/mnt/ramdisk/max/90kDICT32px/annotation_test.txt",
                'r')
    line = file.readline()
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    while line:
        label_name = line.split('_')
        path_name = line.split()
        path = root_path + path_name[0].lstrip("./")

        if not os.path.exists(path):
            print('%s does not exist' % path)
            continue
        if checkValid:
            if not checkImageIsValid(path):
                print('%s is not a valid image' % path)
                line = file.readline()
                continue
        with open(path, 'rb') as f:
            imageBin = f.read()

        imageKey = "image_{:09d}".format(cnt)
        labelKey = "label_{:09d}".format(cnt)
        cache[imageKey] = imageBin
        cache[labelKey] = (label_name[1]).encode(encoding="utf-8")
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d' % (cnt))
        cnt += 1

        # if (cnt == 10000):
        #     break
        line = file.readline()  # 为什么把这个注释掉？

    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    main("/root/project/data/lmdb_test")
