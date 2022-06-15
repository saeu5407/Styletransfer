import random
import glob
import os
from PIL import Image
from torch.utils.data import DataLoader

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class ClabDataset(DataLoader):
    def __init__(self, dataset_dir, styles, transforms):
        self.dataset_dir = dataset_dir
        self.transforms = transforms
        self.image_path_A = glob.glob(os.path.join(dataset_dir, styles[0] ,'*.png'))
        self.image_path_B = []
        category = glob.glob(os.path.join(dataset_dir, "*"))
        if styles[1] == 'all':
            for cat in category:
                if os.path.basename(cat) == styles[0]:
                    continue
                img_list = glob.glob(cat + '/*.png')
                self.image_path_B.extend(img_list)
            print(len(self.image_path_B))
        else:
            self.image_path_B = glob.glob(dataset_dir + '/' + styles[1] + '/*.png')

        random.seed(50)
        self.image_path_A = random.sample(self.image_path_A, 1000)
        self.image_path_B = random.sample(self.image_path_B, 1000)

    def __len__(self):
        return len(self.image_path_A)

    def __getitem__(self, index_A):
        item_A = self.transforms(Image.open(self.image_path_A[index_A]))
        item_B = self.transforms(Image.open(self.image_path_B[index_A]))

        return [item_A, item_B]


class UnpairedDataset(DataLoader):
    def __init__(self, dataset_dir, styles, transforms):
        self.dataset_dir = dataset_dir
        self.styles = styles
        self.image_path_A = glob.glob(os.path.join(dataset_dir, styles[0]) + "/*")
        self.image_path_B = glob.glob(os.path.join(dataset_dir, styles[1]) + "/*")
        self.transform = transforms

    def __getitem__(self, index_A):
        index_B = random.randint(0, len(self.image_path_B) - 1)

        item_A = self.transform(pil_loader(self.image_path_A[index_A]))
        item_B = self.transform(pil_loader(self.image_path_B[index_B]))

        """
        if item_B.shape[0] != 3:
            zeros = torch.zeros(3, item_B[0].shape[0], item_B[0].shape[1])
            zeros[0] = item_B[0]
            zeros[1] = item_B[0]
            zeros[2] = item_B[0]
            item_B = zeros
        """

        return [item_A, item_B]

    def __len__(self):
        return len(self.image_path_A)