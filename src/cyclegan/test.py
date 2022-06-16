import glob
import torch
from PIL import Image

import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from networks import CycleGANGenerator, CycleGANDiscriminator


def generate_image(input_image, netG_A2B, netG_B2A, device, type='A2B'):

    image2tensor = transforms.ToTensor()
    tensor2image = transforms.ToPILImage()

    image = Image.open(input_image)

    # image2tensor
    image_tensor = F.resize(image, (256, 256))
    image_tensor = image2tensor(image_tensor)
    image_tensor = F.normalize(image_tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    image_tensor = image_tensor.unsqueeze(0)

    # resize image
    image = tensor2image(image_tensor.squeeze(0))

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


def mitty_test(dataset_name, img_path, cat_type=['study', 'phone']):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    num_blocks = 6
    netG_A2B = CycleGANGenerator(num_blocks).to(device)
    netG_B2A = CycleGANGenerator(num_blocks).to(device)
    netD_A = CycleGANDiscriminator().to(device)
    netD_B = CycleGANDiscriminator().to(device)

    if len(glob.glob(os.path.join(os.path.join(os.getcwd().split("src")[0], "checkpoint", dataset_name), '*.pth'))) >= 1:
        print("Use .pth")
        checkpoint = torch.load(glob.glob(os.path.join(os.path.join(os.getcwd().split("src")[0], "checkpoint", dataset_name), '*.pth'))[-1], map_location=device)
        # checkpoint["model"] = {key.replace("module.", ""): value for key, value in checkpoint["model"].items()}
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

    B1, BGA1, BGAGB1 = generate_image(cat1_image[1], netG_A2B, netG_B2A, device, "B2A")
    A1, AGB1, AGBGA1 = generate_image(cat2_image[1], netG_A2B, netG_B2A, device, "A2B")

    B1.save('B1.png')
    BGA1.save('BGA1.png')
    BGAGB1.save('BGAGB1.png')
    A1.save('A1.png')
    AGB1.save('AGB1.png')
    AGBGA1.save('AGBGA1.png')

    """
    plt.figure()
    fig, ax = plt.subplots(2,3)
    ax[0,0].imshow(A1)
    ax[0,1].imshow(AGB1)
    ax[0,2].imshow(AGBGA1)

    ax[1,0].imshow(B1)
    ax[1,1].imshow(BGA1)
    ax[1,2].imshow(BGAGB1)
    plt.show()
    """


if __name__ == "__main__":

    data = 'mitty'

    import os

    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    # param
    if data == 'mitty':
        dataset_name = "image"
        img_path = os.path.join(os.getcwd().split("src")[0], "datasets", "mitty", "image")
        cat_type = ['study', 'phone']
    else:
        dataset_name = "horse2zebra"
        img_path = os.path.join(os.getcwd().split("src")[0], "datasets", "horse2zebra")
        cat_type = ['trainA', 'trainB']

    # test
    mitty_test(dataset_name, img_path, cat_type)