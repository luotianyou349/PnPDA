from matplotlib import pyplot as plt
import torch
import datetime


def feature_show(
    feature, filename="x.png", color="BuGn_r"
):
    f='opencood/pic/'
    feature = torch.mean(feature.detach(), dim=0).to("cpu")
    print(feature.shape)
    # 创建一个图像显示这个张量
    fig, ax = plt.subplots()
    ax.imshow(feature, cmap='magma')
    ax.axis("off")  # 隐藏x轴和y轴标签，仅显示图像
    plt.savefig(f+filename+'.png',dpi=300)
    print(filename)
