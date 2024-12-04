import numpy as np
import pickle
import pyvista as pv
import subprocess
import gc
from mpi4py import MPI


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
):
    comm = MPI.COMM_WORLD

    for idx in range(start_frame, n_frames, comm.size):
        i = idx + comm.rank

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

        sampled = pv.ImageData(dimensions=(n_samples,) * 3, spacing=(1 / n_samples,) * 3)
        zoomed = pv.ImageData(dimensions=(int(n_samples * zoom),) * 3, spacing=(1 / n_samples,) * 3)

        for grid, name in zip([sampled, zoomed], ['', '_zoom']):
            p = pv.Plotter(off_screen=True)
            contours = grid.sample(v, progress_bar=True).contour(
                isosurfaces=[0.3], method='flying_edges', progress_bar=True
            )

            p.add_mesh(contours, opacity=0.5, cmap=['teal'])
            p.remove_scalar_bar()
            p.camera.azimuth += 15

            p.camera.Elevation(0.7)
            plotting_path = './simulation_plots/'

            path = f'{plotting_path}/GS_large_{i:06d}{name}.png'
            p.camera.zoom(1.1)
            p.screenshot(path, window_size=(4096, 4096))
            print(f'Saved {path}', flush=True)


def video(view=None):
    if view is None:
        path = 'simulation_plots/GS_large_%06d.png'
        path_target = 'videos/GS_large.mp4'
    elif view == 'slice':
        path = 'simulation_plots/GS_large_slice_%06d.png'
        path_target = 'videos/GS_large_slice.mp4'
    else:
        path = f'simulation_plots/GS_large_{view}_%06d.png'
        path_target = f'videos/GS_large_{view}.mp4'

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
    parser.add_argument('--zoom', type=float, default=1e-2)
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
            pv.start_xvfb()
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
        for view in [None, 'slice']:  #'xy', 'xz', 'yz']:
            video(view)
    else:
        raise NotImplementedError()
