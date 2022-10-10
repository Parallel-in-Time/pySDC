import cupy
import dask.array as da
import time

from dask.distributed import LocalCluster, Client

# generate chunked dask arrays of mamy numpy random arrays
# rs = da.random.RandomState()
# x = rs.normal(10, 1, size=(5000, 5000), chunks=(1000, 1000))
#
# print(f'{x.nbytes / 1e9} GB of data')
#
# t0 = time.perf_counter()
# (x + 1)[::2, ::2].sum().compute(scheduler='single-threaded')
# print(time.perf_counter() - t0)
#
# t0 = time.perf_counter()
# (x + 1)[::2, ::2].sum().compute(scheduler='threads')
# print(time.perf_counter() - t0)

if __name__ == '__main__':
    c = LocalCluster(n_workers=2, processes=True, threads_per_worker=24)
    print(c)

    c = Client()
    print(c)
