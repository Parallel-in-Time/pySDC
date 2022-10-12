import json
import glob
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from argparse import ArgumentParser

import imageio


def plot_data(path='./data', name='', output='.'):
    """
    Visualization using numpy arrays (written via MPI I/O) and json description

    Produces one png file per time-step, combine as movie via e.g.
      > ffmpeg -i data/name_%08d.png name.mp4

    Args:
        path (str): path to data files
        name (str): name of the simulation (expects data to be in data path)
        output (str): path to output
    """

    json_files = sorted(glob.glob(f'{path}/{name}_*.json'))
    data_files = sorted(glob.glob(f'{path}/{name}_*.dat'))

    for json_file, data_file in zip(json_files, data_files):
        with open(json_file, 'r') as fp:
            obj = json.load(fp)

        index = json_file.split('_')[-1].split('.')[0]
        print(f'Working on step {index}...')

        array = np.fromfile(data_file, dtype=obj['datatype'])
        array = array.reshape(obj['shape'], order='C')

        fig = plt.figure()

        grid = AxesGrid(
            fig, 111, nrows_ncols=(1, 2), axes_pad=0.15, cbar_mode='single', cbar_location='right', cbar_pad=0.15
        )

        im = grid[0].imshow(array[..., 0], vmin=0, vmax=1)
        im = grid[1].imshow(array[..., 1], vmin=0, vmax=1)

        grid[0].set_title(f"Field - Time: {obj['time']:6.4f}")
        grid[1].set_title(f"Temperature - Time: {obj['time']:6.4f}")

        grid[1].yaxis.set_visible(False)

        grid.cbar_axes[0].colorbar(im)

        plt.savefig(f'{output}/{name}_{index}.png', bbox_inches='tight')
        plt.close()


def make_movie(path='./data', name='', output='.'):
    """
    Visualization using numpy arrays (written via MPI I/O) and json description

    Produces one png file per time-step, combine as movie via e.g.
      > ffmpeg -i data/name_%08d.png name.mp4

    Args:
        path (str): path to data files
        name (str): name of the simulation (expects data to be in data path)
        output (str): path to output
    """

    json_files = sorted(glob.glob(f'{path}/{name}_*.json'))
    data_files = sorted(glob.glob(f'{path}/{name}_*.dat'))

    img_list = []
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

    fname = f'{output}/{name}.mp4'
    imageio.mimsave(fname, img_list, fps=8)


def make_movie_from_files(path='./data', name='', output='.'):
    """
    Visualization using numpy arrays (written via MPI I/O) and json description

    Produces one png file per time-step, combine as movie via e.g.
      > ffmpeg -i data/name_%08d.png name.mp4

    Args:
        path (str): path to data files
        name (str): name of the simulation (expects data to be in data path)
        output (str): path to output
    """

    img_files = sorted(glob.glob(f'{path}/{name}_*.png'))
    print(f'{path}{name}')

    images = []
    for fimg in img_files:
        img = imageio.imread(fimg)
        print(fimg, img.shape)
        images.append(imageio.imread(fimg))
    fname = f'{output}/{name}.mp4'
    imageio.mimsave(fname, images, fps=8)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-p", "--path", help='Path to data files', type=str, default='./data')
    parser.add_argument("-n", "--name", help='Name of the simulation', type=str)
    parser.add_argument("-o", "--output", help='Path for output file', type=str, default='.')
    args = parser.parse_args()

    # name = 'AC-test-tempforce'
    name = 'AC-bench-tempforce'

    plot_data(path=args.path, name=args.name, output=args.output)
    # make_movie(path=args.path, name=args.name, output=args.output)
    make_movie_from_files(path=args.path, name=args.name, output=args.output)
