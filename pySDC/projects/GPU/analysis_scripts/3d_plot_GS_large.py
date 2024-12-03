import numpy as np
import pickle
import pyvista as pv
import subprocess
import gc
from mpi4py import MPI


def plot(n_time, n_space, useGPU, n_frames, base_path, space_range, res, start_frame=0, zoom=[1, 1, 1], origin=[0,0,0]):
    comm = MPI.COMM_WORLD

    for idx in range(start_frame, n_frames, comm.size):
        p = pv.Plotter(off_screen=True)
        i = idx + comm.rank

        _v = None
        v = None
        gc.collect()
        for procs in space_range:
            gc.collect()

            path = f'{base_path}/GrayScottLarge-res_{res}-useGPU_{useGPU}-procs_1_{n_time}_{n_space}-0-{procs}-solution_{i:06d}.pickle'
            with open(path, 'rb') as file:
                _data = pickle.load(file)

            # if _v is None:
            # _v = np.zeros(_data['shape'])
            if v is None:
                # shape = [int((_data['shape'][i] * zoom[i]) // 1) for i in range 3]
                shape = _data['shape']
                v = pv.ImageData(dimensions=shape)
                v['values'] = np.zeros(np.prod(shape))

            # local_slice_flat = slice(np.prod(shape) * procs, np.prod(shape) * (procs + 1))
            local_slice_flat = slice(np.prod(_data['v'].shape) * procs, np.prod(_data['v'].shape) * (procs + 1))
            # local_slice = [slice(origin[i] 
            v['values'][local_slice_flat] = _data['v'].flatten()
            # _v[*_data['local_slice']] = _data['v']

            # _v[*_data['local_slice']] = _data['v']
            # print(f'loaded data from task {procs} with slice {_data["local_slice"]}', flush=True)

        # print('finished loading data', flush=True)
        # # plot slice
        # v_slice = pv.wrap(_v[:60, ...])
        # print('wrapped data', flush=True)
        # contours = v_slice.contour(isosurfaces=[0.3])
        # p.add_mesh(contours, opacity=0.7, cmap=['teal'])
        # p.remove_scalar_bar()
        # p.camera.azimuth += 15

        # p.camera.Elevation(0.7)
        # plotting_path = './simulation_plots/'

        # path = f'{plotting_path}/GS_large_slice_{i:06d}.png'
        # p.camera.tight(view='yz')
        # p.screenshot(path, window_size=(4096, 4096))
        # print(f'Saved {path}', flush=True)

        # plot whole thing
        contours = v.contour(isosurfaces=[0.3], method='flying_edges', progress_bar=True)
        print('done with contour', flush=True)
        p.add_mesh(contours, opacity=0.7, cmap=['teal'])
        print('added mesh', flush=True)
        p.remove_scalar_bar()
        print('removed scalar bar', flush=True)
        p.camera.azimuth += 15
        print('azimuth', flush=True)

        p.camera.Elevation(0.7)
        print('elevation', flush=True)
        plotting_path = './simulation_plots/'

        path = f'{plotting_path}/GS_large_{i:06d}.png'
        p.camera.zoom(1.1)
        print('zoom', flush=True)
        p.screenshot(path, window_size=(4096, 4096))
        print(f'Saved {path}', flush=True)

        # for view in ['xy', 'xz', 'yz']:
        #     path = f'{plotting_path}/GS_large_{view}_{i:06d}.png'
        #     p.camera.tight(view=view)
        #     p.screenshot(path, window_size=(4096, 4096))
        #     print(f'Saved {path}', flush=True)

        continue
        # plot slice
        v_slice = pv.wrap(_v[:10, ...])
        contours = v_slice.contour(isosurfaces=[0.3], method='flying_edges', progress_bar=True)
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
    parser.add_argument('--space_range', type=int, default=None)
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
        )
    elif args.mode == 'video':
        for view in [None, 'slice']:  #'xy', 'xz', 'yz']:
            video(view)
    else:
        raise NotImplementedError()
