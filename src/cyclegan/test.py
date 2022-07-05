import argparse
import os
import glob
import torch
from PIL import Image

import pandas as pd
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from networks import CycleGANGenerator, CycleGANDiscriminator


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default=os.path.join(os.getcwd().split(os.sep + "src")[0], "datasets", "apple2orange"))
parser.add_argument("--class_a", type=str, default='testA')
parser.add_argument("--class_b", type=str, default='testB')
parser.add_argument("--idx", type=int, default=2)
args = parser.parse_args()


def generate_image(input_image, netG_A2B, netG_B2A, device, type='A2B'):

    image2tensor = transforms.ToTensor()
    tensor2image = transforms.Compose(
        [
            transforms.Normalize(mean=(-1, -1, -1), std=(2, 2, 2)),
            transforms.ToPILImage(),
        ]
    )

    image = Image.open(input_image)

    # image2tensor
    image = F.resize(image, (256, 256))
    image_tensor = image2tensor(image)
    image_tensor = F.normalize(image_tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    image_tensor = image_tensor.unsqueeze(0)

    # model predict
    image_tensor = image_tensor.to(device)
    if type == 'A2B':
        output = netG_A2B(image_tensor)
        output2 = netG_B2A(output)
    else:
        output = netG_B2A(image_tensor)
        output2 = netG_A2B(output)
    generate_image = tensor2image(output.squeeze(0))
    regenerate_image = tensor2image(output2.squeeze(0))

    return image, generate_image, regenerate_image


def cyclegan_test(dataset_name, img_path, cat_type=['study', 'phone'], idx=1):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    num_blocks = 6
    netG_A2B = CycleGANGenerator(num_blocks).to(device)
    netG_B2A = CycleGANGenerator(num_blocks).to(device)
    netD_A = CycleGANDiscriminator().to(device)
    netD_B = CycleGANDiscriminator().to(device)

    checkpoint_path = glob.glob(os.path.join(os.getcwd().split(os.sep + "src")[0], "checkpoint", dataset_name, '*.pth'))
    checkpoint_path = pd.DataFrame({'path' : checkpoint_path, 'name' : list(map(lambda x : os.path.basename(x), checkpoint_path))})
    checkpoint_path.sort_values('name', ascending=True, inplace=True)
    checkpoint_path = list(checkpoint_path.path)

    if len(checkpoint_path) >= 1:
        print("Use {}".format(os.path.basename(checkpoint_path[-1])))
        checkpoint = torch.load(checkpoint_path[-1], map_location=device)
        netG_A2B.load_state_dict(checkpoint["netG_A2B_state_dict"])
        netG_B2A.load_state_dict(checkpoint["netG_B2A_state_dict"])
        netD_A.load_state_dict(checkpoint["netD_A_state_dict"])
        netD_B.load_state_dict(checkpoint["netD_B_state_dict"])

    netG_A2B.eval()
    netG_B2A.eval()
    netD_A.eval()
    netD_B.eval()

    cat1_image = glob.glob(os.path.join(img_path, cat_type[0], '*'))
    cat2_image = glob.glob(os.path.join(img_path, cat_type[1], '*'))

    A1, AGB1, AGBGA1 = generate_image(cat1_image[idx], netG_A2B, netG_B2A, device, "A2B")
    B1, BGA1, BGAGB1 = generate_image(cat2_image[idx], netG_A2B, netG_B2A, device, "B2A")

    A1.save(os.path.join(os.getcwd().split(os.sep + "src")[0], "datasets", dataset_name, 'A_original.png'))
    AGB1.save(os.path.join(os.getcwd().split(os.sep + "src")[0], "datasets", dataset_name, 'A2B_generate.png'))
    AGBGA1.save(os.path.join(os.getcwd().split(os.sep + "src")[0], "datasets", dataset_name, 'A2B2A_generate.png'))
    B1.save(os.path.join(os.getcwd().split(os.sep + "src")[0], "datasets", dataset_name, 'B_original.png'))
    BGA1.save(os.path.join(os.getcwd().split(os.sep + "src")[0], "datasets", dataset_name, 'B2A_generate.png'))
    BGAGB1.save(os.path.join(os.getcwd().split(os.sep + "src")[0], "datasets", dataset_name, 'B2A2B_generate.png'))

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(15,10))
    ax1.imshow(A1)
    ax2.imshow(AGB1)
    ax3.imshow(AGBGA1)

    ax4.imshow(B1)
    ax5.imshow(BGA1)
    ax6.imshow(BGAGB1)
    plt.show()


if __name__ == "__main__":

    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    dataset_name = os.path.basename(args.dataset_path)
    class_path = [args.class_a, args.class_b]

    # test
    cyclegan_test(dataset_name=dataset_name, img_path=args.dataset_path, cat_type=class_path, idx=args.idx)