import os.path

import matplotlib.pyplot as plt
import numpy as np

def make_img(title,y1_name,y2_name,y1,y2,y1_color = 'gray',y2_color='blue',show = False,save_path=None,save_name=None):
    x = np.arange(len(y1))

    fig,ax1 = plt.subplots()

    plt.title(title)

    ax1.plot(x,y1,y1_color, label=y1_name)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel(y1_name,color=y1_color)
    ax1.tick_params('y',color=y1_color)

    ax2 = ax1.twinx()
    ax2.plot(x,y2,y2_color, label=y2_name)
    ax2.set_ylabel(y2_name,color=y2_color)
    ax2.tick_params('y',color=y2_color)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    if save_path and save_name:
        plt.savefig(os.path.join(save_path,save_name))

    if show:
        plt.show()


if __name__ == '__main__':
    # 生成示例数据
    x = np.arange(1000)
    y1 = x ** 3
    y2 = x ** 2

    make_img('test','y1','y2',y1,y2,show=True)