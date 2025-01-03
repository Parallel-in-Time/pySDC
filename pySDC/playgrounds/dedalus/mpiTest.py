from time import sleep
from pySDC.playgrounds.dedalus.sdc import initSpaceTimeMPI

gComm, sComm, tComm = initSpaceTimeMPI(nProcTime=4)

gRank = gComm.Get_rank()
gSize = gComm.Get_size()

sRank = sComm.Get_rank()
sSize = sComm.Get_size()

tRank = tComm.Get_rank()
tSize = tComm.Get_size()

sleep(gRank*0.01)
print(f"Rank {gRank}/{gSize} : sRank {sRank}/{sSize}, tRank {tRank}/{tSize}")