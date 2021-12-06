import torch
import torchvision
import sys
import numpy as np
import lmdb
import six
from PIL import Image


class lmdbDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root=None,
                 transform=None,
                 target_transform=None,
                 imgH=32,
                 imgW=100):
        self.env = lmdb.open(root,
                             max_readers=1,
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get("num-samples".encode()))
            self.nSamples = nSamples

        self.transform = transform
        self.target_transform = target_transform
        self.imgH = imgH
        self.imgW = imgW

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = "image_{:09d}".format(index + 1).encode(encoding="utf-8")
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = torchvision.transforms.functional.to_tensor(
                    Image.open(buf).convert('L').resize(
                        (self.imgW, self.imgH)))
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            if self.transform is not None:
                img = self.transform(img)

            label_key = "label_{:09d}".format(index +
                                              1).encode(encoding="utf-8")
            label = txn.get(label_key).decode()

            if self.target_transform is not None:
                label = self.target_transform(label)

        return img, label


# class synthdataset(torch.utils.data.Dataset):
#     def __init__(self, train=True):
#         # self.root = root
#         self.train = train

#         # if self.train:
#         datalmdb_path = "/root/project/data/lmdb_train1"

#         self.data_env = lmdb.open(datalmdb_path, readonly=True)

#         # else:
#         #     datalmdb_path = 'testdata_lmdb'
#         #     labellmdb_path = 'testlabel_lmdb'
#         #     self.data_env = lmdb.open(datalmdb_path, readonly=True)
#         #     self.label_env = lmdb.open(labellmdb_path, readonly=True)

#     def __getitem__(self, index):
#         # assert index == 12

#         Data = []
#         Target = []
#         imageKey = "image_{:09d}".format(index + 1).encode(encoding="utf-8")
#         labelKey = "label_{:09d}".format(index + 1).encode(encoding="utf-8")

#         # if self.train:
#         with self.data_env.begin() as f:
#             data = f.get(imageKey)
#             # buf=six.BytesIO()
#             # buf.write(data)
#             # buf.seek(0)
#             # # try:
#             # data=Image.open(data).convert('RGB')

#             data = np.frombuffer(data, dtype=np.uint8)
#             data = cv2.imdecode(data, cv2.IMREAD_COLOR)
#             cv2.resize(data, (30, 80))
#             data = np.array(data)

#             # except IOError:
#             #     print(index)
#             #     return self[index+1]
#             # if data is not None:
#             #     flat_data = np.frombuffer(data, dtype=float)
#             #     data = flat_data.reshape(150, 6).astype('float32')
#             # else:
#             #     data=np.ones((150,6),dtype=float32)

#             # Data = np.array(data)
#             Data = data

#         with self.data_env.begin() as f:
#             data = f.get(labelKey)

#             Target = data.decode()

#         # else:

#         #     with self.data_env.begin() as f:
#         #         key = '{:08}'.format(index)
#         #         data = f.get(key)
#         #         flat_data = np.fromstring(data, dtype=float)
#         #         data = flat_data.reshape(150, 6).astype('float32')
#         #         Data = data

#         #     with self.label_env.begin() as f:
#         #         key = '{:08}'.format(index)
#         #         data = f.get(key)
#         #         label = np.fromstring(data, dtype=int)
#         #         Target = label[0]

#         return Data, Target

#     def __len__(self):

#         return 10

#         # if self.train:
#         #     return 2693931
#         # else:
#         #     return 224589

# mya = lmdbDataset(root="/root/project/data/lmdb_train1")
# thloard = torch.utils.data.DataLoader(dataset=mya, batch_size=3, shuffle=True)
# for img, target in thloard:
#     print(target)
