import numpy as np
import pickle
import pyvista as pv
import subprocess
import gc
from mpi4py import MPI
from tqdm import tqdm


class Grid:
    def __init__(self, grid, label, zoom_range):
        self.grid = grid
        self.label = label
        self.zoom_range = zoom_range

    def get_camera_path(self):
        import os

        os.makedirs('etc/cameras', exist_ok=True)
        return f'etc/cameras/cam{self.label}_{len(self.grid.x)}.pickle'

    def get_camera_pos(self):
        with open(self.get_camera_path(), 'rb') as file:
            return pickle.load(file)

    def set_camera_pos(self, pos):
        with open(self.get_camera_path(), 'wb') as file:
            pickle.dump(pos, file)

    def get_zoom(self, frame):
        return (self.zoom_range[1] - self.zoom_range[0]) * frame / 100 + self.zoom_range[0]


def plot(
    n_time,
    n_space,
    useGPU,
    n_frames,
    base_path,
    space_range,
    res,
    start_frame=0,
    plot_resolution=2048,
    n_samples=1024,
    zoom=1e-2,
):  # pragma: no cover
    comm = MPI.COMM_WORLD

    space_range = tqdm(space_range)
    space_range.set_description('load files')

    for frame in range(start_frame, n_frames, comm.size):
        i = frame + comm.rank

        v = None
        gc.collect()
        for procs in space_range:
            gc.collect()

            path = f'{base_path}/GrayScottLarge-res_{res}-useGPU_{useGPU}-procs_1_{n_time}_{n_space}-0-{procs}-solution_{i:06d}.pickle'
            with open(path, 'rb') as file:
                _data = pickle.load(file)

            if v is None:
                shape = _data['shape']
                v = pv.ImageData(dimensions=shape, spacing=tuple([1 / me for me in shape]))
                v['values'] = np.zeros(np.prod(shape))

            local_slice_flat = slice(np.prod(_data['v'].shape) * procs, np.prod(_data['v'].shape) * (procs + 1))
            v['values'][local_slice_flat] = _data['v'].flatten()

        # sampled = Grid(pv.ImageData(dimensions=(n_samples,) * 3, spacing=(1 / n_samples,) * 3), '', [1.0, 1.0])
        zoomed = Grid(
            pv.ImageData(dimensions=(n_samples,) * 3, spacing=(zoom / n_samples,) * 3, origin=[0.8, 0.47, 0.03]),
            '_zoomed',
            [1.0, 1.2],
        )

        for grid in [zoomed]:

            p = pv.Plotter(off_screen=True)
            contours = grid.grid.sample(v, progress_bar=True, categorical=True).contour(
                isosurfaces=[0.3], method='flying_edges', progress_bar=True
            )

            p.add_mesh(contours, opacity=0.5, cmap=['teal'])
            p.remove_scalar_bar()
            p.camera.azimuth += 15

            p.camera.Elevation(0.9)
            plotting_path = './simulation_plots/'

            path = f'{plotting_path}/GS_large_{i:06d}{grid.label}.png'

            if frame == 0:
                grid.set_camera_pos(p.camera.position)

            p.camera.position = tuple(pos * grid.get_zoom(frame) for pos in grid.get_camera_pos())

            p.screenshot(path, window_size=(plot_resolution,) * 2)
            print(f'Saved {path}', flush=True)


def video(view=None):  # pragma: no cover
    path = f'simulation_plots/GS_large_%06d{view}.png'
    path_target = f'videos/GS_large{view}.mp4'

    cmd = f'ffmpeg -i {path} -pix_fmt yuv420p -r 9 -s 2048:1536 -y {path_target}'.split()

    subprocess.run(cmd)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--XPU', type=str, choices=['CPU', 'GPU'], default='GPU')
    parser.add_argument('--mode', type=str, choices=['plot', 'video'], default='plot')
    parser.add_argument('--nframes', type=int, default=100)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--base_path', type=str, default='/p/scratch/ccstma/baumann7/large_runs/data/')
    parser.add_argument('--space_range', type=int, default=None)
    parser.add_argument('--zoom', type=float, default=9e-2)
    parser.add_argument('--n_samples', type=int, default=1024)
    args = parser.parse_args()

    from pySDC.projects.GPU.analysis_scripts.large_simulations import GSLarge

    sim = GSLarge()

    if args.XPU == 'CPU':
        sim.setup_CPU_params()
    elif args.XPU == 'GPU':
        sim.setup_GPU_params()
    else:
        raise NotImplementedError()

    space_range = range(sim.params['procs'][2] if args.space_range is None else args.space_range)

    if args.mode == 'plot':
        pv.global_theme.allow_empty_mesh = True
        try:
            pv.start_xvfb(window_size=(4096, 4096))
        except OSError:
            pass
        plot(
            n_time=sim.params['procs'][1],
            n_space=sim.params['procs'][2],
            useGPU=sim.params['useGPU'],
            base_path=args.base_path,
            space_range=space_range,
            n_frames=args.nframes,
            res=sim.params['res'],
            start_frame=args.start,
            n_samples=args.n_samples,
            zoom=args.zoom,
        )
    elif args.mode == 'video':
        for view in ['', '_zoomed']:
            video(view)
    else:
        raise NotImplementedError()
