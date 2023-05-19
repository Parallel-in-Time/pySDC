# python class that calls c functions to handle compression/decompression
import numpy as np

np.bool = np.bool_
import libpressio
from pySDC.implementations.datatype_classes.mesh import mesh


class CRAM_Manager:
    # constructor
    def __init__(self, errBoundMode="ABS", compType="sz3", errBound=1e-5):
        # print("constructor called!")
        # if self.init = 0
        self.errBoundMode = errBoundMode
        self.errBound = errBound
        self.compType = compType
        self.mem_map = {}
        self.cache = {}
        self.max_cache_size = 30
        # self.init = 1
        self.name = 0  # TODO: update registration to return name

    # destructor
    def __del__(self):
        pass

    # numVectors is M
    def registerVar(
        self,
        varName,
        shape,
        dtype=np.dtype("float64"),
        numVectors=1,
        errBoundMode=None,
        compType=None,
        errBound=None,
    ):
        if varName not in self.mem_map:
            # print("Register: ", varName, "-", shape, len(self.mem_map.keys()))
            compressor = libpressio.PressioCompressor.from_config(
                self.generate_compressor_config(compType, errBoundMode, errBound)
            )

            self.mem_map[varName] = [
                compressor,
                [compressor.encode(np.ones(shape)) for i in range(numVectors)],
                shape,
            ]  # TODO finish

    def remove(self, name, index):
        self.cache.pop(name + "_" + str(index), None)
        self.mem_map.pop(name, None)

    def compress(
        self, data, varName, index, errBoundMode="ABS", compType="sz3", errBound=None
    ):
        # print("Cprss: ", varName, index)
        # print("Array: ", data)
        # print("Error bound: ",errBound)
        if errBound is not None:
            compressor = libpressio.PressioCompressor.from_config(
                self.generate_compressor_config(compType, errBoundMode, errBound)
            )
            # cfg = compressor.get_config()
            # print(errBound)
            # cfg['compressor_config']['pressio:abs'] = errBound

            # compressor.set_config(cfg)
            self.mem_map[varName][0] = compressor
        else:
            compressor = self.mem_map[varName][0]
        # print(compressor.get_config()['compressor_id'])
        # print(compressor.get_config()['compressor_config'])
        self.mem_map[varName][1][index] = compressor.encode(data)
        self.cache.pop(self.cacheName(varName, index), None)

    def cacheName(self, varName, index):
        return varName + "_" + str(index)

    def decompress(self, varName, index, compType=0):
        # print("Decprss: ", varName, index)
        combineName = self.cacheName(varName, index)
        if combineName not in self.cache:
            compressor = self.mem_map[varName][0]
            comp_data = self.mem_map[varName][1][index]
            decomp_data = np.zeros(self.mem_map[varName][2])
            # if comp_data != None:
            tmp = compressor.decode(comp_data, decomp_data)

            if (
                len(self.cache) + 1 > self.max_cache_size
            ):  # TODO: Add LRU replacement policy
                k = list(self.cache.keys())[0]
                self.cache.pop(k)
            self.cache[combineName] = tmp
            return tmp

            # else:
            #    tmp = decomp_data

            tmp_mesh = mesh(self.mem_map[varName][2])
            tmp_mesh[:] = tmp

        else:
            # print ("Found in Cache")
            return self.cache[combineName]

    def set_global_compressor_config(
        self, compType=None, errBoundMode=None, errBound=None
    ):
        self.compType = self.compType if compType is None else compType
        self.errBoundMode = self.errBoundMode if errBoundMode is None else errBoundMode
        self.errBound = self.errBound if errBound is None else errBound
        for k in self.mem_map:
            self.mem_map[k][0] = libpressio.PressioCompressor.from_config(
                self.generate_compressor_config(
                    self.compType, self.errBoundMode, self.errBound
                )
            )

    def generate_compressor_config(
        self, compType=None, errBoundMode=None, errBound=None
    ):
        compType = self.compType if compType is None else compType
        errBoundMode = self.errBoundMode if errBoundMode is None else errBoundMode
        errBound = self.errBound if errBound is None else errBound
        # print('Error bound: ',errBound)
        return {
            # configure which compressor to use
            "compressor_id": compType,
            # configure the set of metrics to be gathered
            "early_config": {
                "pressio:metric": "composite",
                "composite:plugins": ["time", "size", "error_stat"],
            },
            # configure SZ
            "compressor_config": {
                "pressio:abs": errBound,
            },
        }

    def __str__(self):
        # print values in the dictionary
        s = "Memory\n"
        for k in self.mem_map.keys():
            s += k + str(self.mem_map[k]) + "\n"

        s += "\nCache\n"
        for k in self.cache.keys():
            s += k + str(self.cache[k]) + "\n"

        return s

    # def printStats(self, varName, index=0):
    #    print(" ")  #for readability
    #   sz_class.sz_printVarInfo(varName)

    def getStats(self, varName, index=0):
        compressor = self.mem_map[varName][0]
        return compressor.get_metrics()


if __name__ == "__main__":
    arr = np.random.rand(100, 100)
    # declare global instance of memory
    memory = CRAM_Manager(errBoundMode="ABS", compType="sz3", errBound=1e-5)

    memory.registerVar(
        "cat",
        arr.shape,
        dtype=np.dtype("float64"),
        numVectors=1,
        errBoundMode="ABS",
        compType="sz3",
        errBound=0.1,
    )
    memory.compress(arr, "cat", 0)
    result = memory.decompress("cat", 0, shape=arr.shape)
    print(arr)
    print("\n")
    print(result)
    print("\n")
    error = arr - result
    print(error)
