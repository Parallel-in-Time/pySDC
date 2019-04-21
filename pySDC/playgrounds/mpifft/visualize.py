import json
import glob
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


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

        plt.imshow(array, vmin=0, vmax=1)

        plt.colorbar()
        plt.title(f"Time: {obj['time']:6.4f}")

        plt.savefig(f'data/{name}_{index}.png', rasterized=True, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":

    # name = 'AC-test'
    # name = 'AC-test-constforce'
    # name = 'AC-2D-application'
    name = 'AC-2D-application-forced'

    plot_data(name=name)
