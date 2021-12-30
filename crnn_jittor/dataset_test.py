import jittor
from jittor import transform
import sys
import lmdb
import six
from PIL import Image


class lmdbTestDataset(jittor.dataset.Dataset):
    def __init__(self,
                 root=None,
                 transform=None,
                 target_transform=None,
                 imgH=32,
                 imgW=100):
        super().__init__()
        self.env = lmdb.open(root,
                             max_readers=10,
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get("num-samples".encode()))
            self.nSamples = nSamples  # 7,224,586

        self.transform = transform
        self.target_transform = target_transform
        self.imgH = imgH  # 32
        self.imgW = imgW  # 100

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        with self.env.begin(write=False) as txn:
            img_key = "image_{:09d}".format(index + 1).encode(encoding="utf-8")
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = transform.to_tensor(
                    Image.open(buf).convert('L').resize(
                        (self.imgW, self.imgH)))
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            if self.transform is not None:
                img = self.transform(img)
            # print("img.shape ", img.shape)

            label_key = "label_{:09d}".format(index + 1).encode(encoding="utf-8")
            label = txn.get(label_key).decode()

            if self.target_transform is not None:
                label = self.target_transform(label)
            # print("label ", label)

            path_key = "path_{:09d}".format(index + 1).encode(encoding="utf-8")
            path = txn.get(path_key).decode()

        return img, label, path  # [1, 32, 100], str, str
