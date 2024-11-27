import numpy as np
import pickle
import pyvista as pv
import subprocess

n_time = 1
n_space = 4
res = 64
useGPU = 'False'
n_frames = 30
base_path = './data'

space_range = range(n_space // 4, n_space // 3)


for i in range(n_frames):
    p = pv.Plotter(off_screen=True)

    v = None
    for procs in space_range:

        path = f'base_path/GrayScottLarge-res_{res}-useGPU_{useGPU}-procs_1_{n_time}_{n_space}-{n_time-1}-{procs}-solution_{i:06d}.pickle'
        with open(path, 'rb') as file:
            _data = pickle.load(file)

        if v is None:
            v = np.zeros(_data['shape'])

        v[*_data['local_slice']] = _data['v']
        # v.origin += np.array([procs * data['v'].shape[0], 0, 0])

    v = pv.wrap(v)

    contours = v.contour(isosurfaces=[0.3])
    p.add_mesh(contours, opacity=0.6)
    # p.add_mesh(v.outline(), color="k")
    p.remove_scalar_bar()

    # p.camera_position = [
    #     (-130.99381142132086, 644.4868354828589, 163.80447435848686),
    #     (125.21748748157661, 123.94368717158413, 108.83283586619626),
    #     (0.2780372840777734, 0.03547871361794171, 0.9599148553609699),
    # ]
    p.show()
    path = f'./simulation_plots/GS_large_{i:06d}.png'
    p.screenshot(path, window_size=(4096, 4096))
    print(f'Saved {path}', flush=True)
print('done')


path = 'simulation_plots/GS_large_%06d.png'
path_target = 'videos/GS_large.mp4'

cmd = f'ffmpeg -i {path} -pix_fmt yuv420p -r 9 -s 2048:1536 -y {path_target}'.split()

subprocess.run(cmd)
