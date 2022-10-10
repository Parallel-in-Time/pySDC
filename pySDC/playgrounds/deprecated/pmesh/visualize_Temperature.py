import json
import glob
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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

        plt.figure()

        plt.imshow(array[..., 0], vmin=0, vmax=1)

        plt.colorbar()
        plt.title(f"Field - Time: {obj['time']:6.4f}")

        plt.savefig(f'data/{name}_field_{index}.png', bbox_inches='tight')
        plt.close()

        plt.figure()

        plt.imshow(array[..., 1], vmin=0, vmax=1)

        plt.colorbar()
        plt.title(f"Temperature - Time: {obj['time']:6.4f}")

        plt.savefig(f'data/{name}_temperature_{index}.png', bbox_inches='tight')
        plt.close()


def make_gif(name=''):
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

        ax[0].imshow(array[..., 1], vmin=0, vmax=1)
        ax[1].imshow(array[..., 0], vmin=0, vmax=1)

        # ax.set_colorbar()
        ax[0].set_title(f"Temperature - Time: {obj['time']:6.4f}")
        ax[1].set_title(f"Field - Time: {obj['time']:6.4f}")

        fig.tight_layout()

        fig.canvas.draw()  # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        img_list.append(image.reshape(fig.canvas.get_width_height()[::-1] + (3,)))
        plt.close()

        # c +=1
        # if c == 3:
        #     break

    # imageio.mimsave('./test.gif', img_list, fps=8, subrectangles=True)
    imageio.mimsave('./test.mp4', img_list, fps=8)


if __name__ == "__main__":

    # name = 'AC-test'
    name = 'AC-temperature-test'
    # name = 'AC-2D-application'
    # name = 'AC-2D-application-forced'

    # plot_data(name=name)
    make_gif(name=name)
