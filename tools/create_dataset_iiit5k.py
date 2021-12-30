import os
import lmdb  # install lmdb by "pip install lmdb"
from PIL import Image

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

def main(outputPath, checkValid=True):

    root_path = "/root/iiit5k/IIIT5K/"

    file = open("/root/iiit5k/IIIT5K/test_annotations.txt", 'r')  # 这里改是 test 还是 train
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    line = file.readline()
    while line:
        path_name = line.split()
        path_key = path_name[0].split('/')[1].split('.')[0]
        img_path = root_path + path_name[0]
        label = path_name[1]

        if not os.path.exists(img_path):
            print('%s does not exist' % img_path)
            continue
        if checkValid:
            if not checkImageIsValid(img_path):
                print('%s is not a valid image' % img_path)
                line = file.readline()
                continue
        with open(img_path, 'rb') as f:
            imageBin = f.read()

        imageKey = "image_{:09d}".format(cnt)
        labelKey = "label_{:09d}".format(cnt)
        pathKey = "path_{:09d}".format(cnt)
        cache[imageKey] = imageBin
        cache[labelKey] = (label).encode(encoding="utf-8")
        cache[pathKey] = (path_key).encode(encoding="utf-8")
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d' % (cnt))
        cnt += 1
        line = file.readline()

    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    main("/root/project/data/lmdb_iiit5k_test")  # 这里改是 test 还是 train
