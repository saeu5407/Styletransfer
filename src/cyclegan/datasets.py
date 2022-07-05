import random
import glob
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class CycleGANDataset(DataLoader):
    def __init__(self, dataset_dir, styles, transforms, random_mode=False):
        """
        CycleGAN 전용으로 만들어 둔 클래스입니다.
        데이터셋 경로 내에 클래스 별 폴더를 만들어 이미지를 넣어 두시고, styles에 StyleTransfer를 원하는 클래스 리스트를 적으시면 됩니다.

        :param dataset_dir: 데이터 셋의 경로를 입력합니다.
        :param styles: 데이터 셋에서 이미 나누어 놓은 하위 디렉토리 리스트를 입력합니다.
                       예를 들어 dog를 cat로 transfer한다면 데이터셋 경로에 dog 폴더와 cat 폴더를 생성한 후,
                       각각의 이미지를 넣고나서 ["dog", "cat"] 리스트를 기입합니다.

                       두 번째 값에 "all"을 적을 시 첫 번째 값 외의 모든 폴더를 사용하게 됩니다.
                       ["dog", "all"] 일 경우, 해당 데이터셋에서 dog 외의 모든 폴더를 사용합니다.
        :param transforms: 미리 설정한 transforms을 입력합니다.
        """
        self.dataset_dir = dataset_dir
        self.transforms = transforms
        self.image_path_A = glob.glob(os.path.join(dataset_dir, styles[0] ,'*')) # styles[0] : A
        self.image_path_B = []
        self.random_mode = random_mode # 클래스를 인덱스 1:1 대응할지 B를 랜덤셔플하여 대응할지 여부

        # styles[1]이 "all"일 때는 style[0] 외의 모든 폴더의 이미지를 보게끔 설정
        if styles[1] == 'all':
            category = glob.glob(os.path.join(dataset_dir, "*"))

            for cat in category:
                if os.path.basename(cat) == styles[0]:
                    continue
                img_list = glob.glob(cat + '/*')
                self.image_path_B.extend(img_list) # style[0] 외의 모든 이미지 extend
        else:
            self.image_path_B = glob.glob(dataset_dir + '/' + styles[1] + '/*')

        # 길이를 맞추어 셔플
        random.seed(50)
        len_a = len(self.image_path_A)
        self.len_image = len(self.image_path_B) if len_a >= len(self.image_path_B) else len(self.image_path_A)
        self.image_path_A = random.sample(self.image_path_A, self.len_image)
        self.image_path_B = random.sample(self.image_path_B, self.len_image)
        print("data length : {}".format(self.len_image))

    def __len__(self):
        return self.len_image

    def __getitem__(self, idx):

        item_a = self.transforms(Image.open(self.image_path_A[idx]))
        if self.random_mode: # random_mode시 item_b를 인덱스 말고 랜덤하게 추출
            random.seed(50)
            item_b = self.transforms(Image.open(random.sample(self.image_path_B, 1)[0]))
        else:
            item_b = self.transforms(Image.open(self.image_path_B[idx]))

        return [item_a, item_b]