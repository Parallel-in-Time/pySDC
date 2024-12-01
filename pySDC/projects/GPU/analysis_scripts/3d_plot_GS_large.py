import numpy as np
import pickle
import pyvista as pv
import subprocess
import gc


def plot(n_time, n_space, useGPU, n_frames, base_path, space_range, res, start_frame=0):
    for i in range(start_frame, n_frames):
        p = pv.Plotter(off_screen=True)

        _v = None
        gc.collect()
        for procs in space_range:
            gc.collect()

            path = f'{base_path}/GrayScottLarge-res_{res}-useGPU_{useGPU}-procs_1_{n_time}_{n_space}-0-{procs}-solution_{i:06d}.pickle'
            with open(path, 'rb') as file:
                _data = pickle.load(file)

            if _v is None:
                _v = np.zeros(_data['shape'])

            _v[*_data['local_slice']] = _data['v']

        v = pv.wrap(_v)

        # plot whole thing
        contours = v.contour(isosurfaces=[0.3])
        p.add_mesh(contours, opacity=0.7, cmap=['teal'])
        p.remove_scalar_bar()
        p.camera.azimuth += 15

        p.camera.Elevation(0.7)
        plotting_path = './simulation_plots/'

        path = f'{plotting_path}/GS_large_{i:06d}.png'
        p.camera.zoom(1.1)
        p.screenshot(path, window_size=(4096, 4096))
        print(f'Saved {path}', flush=True)

        # for view in ['xy', 'xz', 'yz']:
        #     path = f'{plotting_path}/GS_large_{view}_{i:06d}.png'
        #     p.camera.tight(view=view)
        #     p.screenshot(path, window_size=(4096, 4096))
        #     print(f'Saved {path}', flush=True)

        # plot slice
        v_slice = pv.wrap(_v[:10, ...])
        contours = v_slice.contour(isosurfaces=[0.3])
        p.add_mesh(contours, opacity=0.7, cmap=['teal'])
        p.remove_scalar_bar()
        p.camera.azimuth += 15

        p.camera.Elevation(0.7)
        plotting_path = './simulation_plots/'

        path = f'{plotting_path}/GS_large_slice_{i:06d}.png'
        p.camera.tight(view='yz')
        p.screenshot(path, window_size=(4096, 4096))
        print(f'Saved {path}', flush=True)

    print('done')


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
    args = parser.parse_args()

    from pySDC.projects.GPU.analysis_scripts.large_simulations import GSLarge

    sim = GSLarge()

    if args.XPU == 'CPU':
        sim.setup_CPU_params()
    elif args.XPU == 'GPU':
        sim.setup_GPU_params()
    else:
        raise NotImplementedError()

    if args.mode == 'plot':
        try:
            pv.start_xvfb()
        except OSError:
            pass
        plot(
            n_time=sim.params['procs'][1],
            n_space=sim.params['procs'][2],
            useGPU=sim.params['useGPU'],
            base_path=args.base_path,
            space_range=range(sim.params['procs'][2]),
            n_frames=args.nframes,
            res=sim.params['res'],
            start_frame=args.start,
        )
    elif args.mode == 'video':
        for view in [None, 'slice']:  #'xy', 'xz', 'yz']:
            video(view)
    else:
        raise NotImplementedError()
