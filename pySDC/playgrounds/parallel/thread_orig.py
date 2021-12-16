from mpi4py import MPI
import time
import threading


def _send():
    global time0, time1
    comm.send(data,dest=otherrank,tag=1)
    time1 = time.perf_counter() - time0


comm = MPI.COMM_WORLD

reqlist = []
data = ['myid_particles:'+str(comm.rank)]*10000000
otherrank = 1 if comm.rank==0 else 0

send_thread = threading.Thread(target=_send)
time0 = time1 = time2 = time3 = 0
time0 = time.perf_counter()
send_thread.start()


if comm.rank==1:
   time.sleep(10)
time2 = time.perf_counter() - time0
a = comm.recv(source=otherrank,tag=1)
time3 = time.perf_counter() - time0
send_thread.join()
print(str(comm.rank)+': send at t = '+str(time1))
print(str(comm.rank)+': recv at t = ('+str(time2)+','+str(time3)+')')