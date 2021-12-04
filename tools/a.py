# import cv2

path = "/root/synth/mnt/ramdisk/max/90kDICT32px/"

file = open("/root/synth/mnt/ramdisk/max/90kDICT32px/annotation_val.txt", 'r')
line = file.readline()
cnt = 0
while line:
    # print(line)
    # a1 = line.split('_')
    # a2 = line.split()
    # print(a2[0].lstrip("./"))
    # img = cv2.imread(path + a2[0].lstrip("./"))
    # cv2.imwrite("/root/project/torch/a.png",img)
    # print(a2[0], a1[1])
    cnt += 1
    # if cnt == 10:
    #     break
    line = file.readline()

print(cnt)
# cnt = 21

# imageKey = "image_{:09d}".format(cnt)

# print(imageKey)