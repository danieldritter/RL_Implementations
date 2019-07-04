import Model
import torchvision.datasets as datasets
import torch.utils as utils
import torchvision.transforms as transforms
import PIL.Image as Image

"""
Unit Tests for various pieces of MaskRCNN
"""


def FPN_Tests():
    model = Model.FPN()
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    mnist_dataloader = utils.data.DataLoader(dataset=test_set, batch_size=10, shuffle=True)
    for i, example in enumerate(mnist_dataloader):
        input, target = example
        out = model(input)
        print(out)


def __main__():
    FPN_Tests()


if __name__ == "__main__":
    __main__()
