import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PlotCounter:
    counter = 0

    def get_counter(self):
        PlotCounter.counter += 1
        return PlotCounter.counter

def plot(sp_tensor1, sp_tensor2):
    fig, axes = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(12, 6))
    sp_tensor1.plot_3d(fig, axes[0])
    sp_tensor2.plot_3d(fig, axes[1])
    y_lim = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    z_lim = max(axes[0].get_zlim()[1], axes[1].get_zlim()[1])
    axes[0].set_ylim([0, y_lim])
    axes[0].set_zlim([0, z_lim])
    plt.tight_layout()  
    plt.show()
    # plt.savefig(f'plot{PlotCounter().get_counter()}.png')
    # plt.close()