import matplotlib
import os
if 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')
    show_image = False
else:
    show_image = True
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
import torchvision.utils as vutils


def imshow(inp, file_name='NoName.png', save=False, title=None):
    """Imshow for Tensor."""
    fig = plt.figure(figsize=(5, 5))
    inp = transforms.ToPILImage()(inp)
    plt.imshow(inp)
    plt.savefig(file_name)
    if show_image:
        plt.show()
    plt.gcf().clear()
