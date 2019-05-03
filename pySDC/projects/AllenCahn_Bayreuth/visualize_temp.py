import json
import glob
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

import imageio


def plot_data(name=''):
    """
    Visualization using numpy arrays (written via MPI I/O) and json description

    Produces one png file per time-step, combine as movie via e.g.
      > ffmpeg -i data/name_%08d.png name.mp4

    Args:
        name (str): name of the simulation (expects data to be in data path)
    """

    json_files = sorted(glob.glob(f'./data/{name}_*.json'))
    data_files = sorted(glob.glob(f'./data/{name}_*.dat'))

    for json_file, data_file in zip(json_files, data_files):
        with open(json_file, 'r') as fp:
            obj = json.load(fp)

        index = json_file.split('_')[1].split('.')[0]
        print(f'Working on step {index}...')

        array = np.fromfile(data_file, dtype=obj['datatype'])
        array = array.reshape(obj['shape'], order='C')

        # fig = plt.figure(figsize=(6, 4))
        fig = plt.figure()

        grid = AxesGrid(fig, 111,
                        nrows_ncols=(1, 2),
                        axes_pad=0.15,
                        cbar_mode='single',
                        cbar_location='right',
                        cbar_pad=0.1
                        )

        im = grid[0].imshow(array[..., 0], vmin=0, vmax=1)
        im = grid[1].imshow(array[..., 1], vmin=0, vmax=1)

        grid[0].set_title(f"Field - Time: {obj['time']:6.4f}")
        grid[1].set_title(f"Temperature - Time: {obj['time']:6.4f}")

        grid[1].yaxis.set_visible(False)

        grid.cbar_axes[0].colorbar(im)

        plt.savefig(f'data/{name}_{index}.png', rasterized=True, bbox_inches='tight')
        plt.close()
        # break


def make_movie(name=''):
    """
    Visualization using numpy arrays (written via MPI I/O) and json description

    Produces one png file per time-step, combine as movie via e.g.
      > ffmpeg -i data/name_%08d.png name.mp4

    Args:
        name (str): name of the simulation (expects data to be in data path)
    """

    json_files = sorted(glob.glob(f'./data/{name}_*.json'))
    data_files = sorted(glob.glob(f'./data/{name}_*.dat'))
    img_list = []
    c = 0
    for json_file, data_file in zip(json_files, data_files):
        with open(json_file, 'r') as fp:
            obj = json.load(fp)

        index = json_file.split('_')[1].split('.')[0]
        print(f'Working on step {index}...')

        array = np.fromfile(data_file, dtype=obj['datatype'])
        array = array.reshape(obj['shape'], order='C')

        fig, ax = plt.subplots(1, 2)

        ax[0].imshow(array[..., 0], vmin=0, vmax=1)
        ax[1].imshow(array[..., 1], vmin=0, vmax=1)

        # ax.set_colorbar()
        ax[0].set_title(f"Field - Time: {obj['time']:6.4f}")
        ax[1].set_title(f"Temperature - Time: {obj['time']:6.4f}")

        fig.tight_layout()

        # draw the canvas, cache the renderer
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        img_list.append(image.reshape(fig.canvas.get_width_height()[::-1] + (3,)))
        plt.close()

        # c += 1
        # if c == 3:
        #     break

    fname = f'{name}.mp4'
    imageio.mimsave(fname, img_list, fps=8)


def make_movie_from_files(name=''):
    """
    Visualization using numpy arrays (written via MPI I/O) and json description

    Produces one png file per time-step, combine as movie via e.g.
      > ffmpeg -i data/name_%08d.png name.mp4

    Args:
        name (str): name of the simulation (expects data to be in data path)
    """

    img_files = sorted(glob.glob(f'data/{name}_*.png'))

    images = []
    for fimg in img_files:
        img = imageio.imread(fimg)
        print(fimg, img.shape)
        images.append(imageio.imread(fimg))
    fname = f'{name}.mp4'
    imageio.mimsave(fname, images, fps=8)


if __name__ == "__main__":

    name = 'AC-test-tempforce'
    plot_data(name=name)
    # make_movie(name=name)
    make_movie_from_files(name=name)
