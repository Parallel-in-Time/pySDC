import numpy as np

np.bool = np.bool_
import libpressio
from pySDC.implementations.datatype_classes.mesh import mesh
from pySDC.projects.compression.cache_manager import Cache

from time import time
import os


class CRAM_Manager:
    # Cache attribute:
    cacheManager = Cache()

    # constructor
    # def __init__(self, errBoundMode="PW_REL", compType="sz3", errBound=1e-5):
    def __init__(self, errBoundMode="PW_REL", compType="blosc", errBound=1e-5, losslesscompressor="zstd"):
        # Parameters for the error bound and compressor
        self.errBoundMode = errBoundMode
        self.errBound = errBound
        self.compType = compType
        self.losslesscompressor = losslesscompressor
        # Parameters for memory map and cache (bool to enable logging cache history as well)
        self.mem_map = {}
        self.cacheHist = []
        self.trackBaseline = True
        self.logcacheHist = False
        self.cachePolicy = ['LRU', 'LFU']
        # Parameters to compute and store metrics
        self.registerVars_time = 0
        self.compression_time_nocache = 0  # No cache, compression time
        self.compression_time = 0  # Not found in cache, compression, write back eviction maybe and put time
        self.compression_time_update = 0  # Found in cache, only update decompressed data time
        self.compression_time_eviction = 0  # write back cache eviction time
        self.compression_time_put_only = 0  # compression - put time
        self.decompression_time_nocache = 0  # No cache, decompression time
        self.decompression_time = 0  # Not found in cache, decompression, write back eviction maybe and put time
        self.decompression_time_get = 0  # Found in cache, get (retrieve it)
        self.decompression_time_eviction = 0  # write back cache eviction time
        self.decompression_time_put_only = 0  # decompression - put time
        self.num_registered_var = 0
        self.num_active_registered_var = 0
        self.num_compression_calls = 0
        self.num_decompression_calls = 0
        self.name = 0  # TODO: update registration to return name

    # destructor
    def __del__(self):
        pass

    # Registers array to be compressed, decompressed in memory map if it does not already exists and updates number of registrations
    def registerVar(
        self,
        varName,
        shape,
        dtype=np.dtype("float64"),
        numVectors=1,
        errBoundMode=None,
        compType=None,
        errBound=None,
        losslesscompressor=None,
    ):
        start_time = time()
        if varName not in self.mem_map:
            # print("Register: ", varName, "-", shape, len(self.mem_map.keys()))
            compressor = libpressio.PressioCompressor.from_config(
                self.generate_compressor_config(compType, errBoundMode, errBound, losslesscompressor)
            )

            self.mem_map[varName] = [
                compressor,
                [compressor.encode(np.ones(shape)) for i in range(numVectors)],
                shape,
            ]  # TODO finish

            self.num_registered_var += 1
            self.num_active_registered_var += 1
        end_time = time()
        time_diff = end_time - start_time
        self.registerVars_time += time_diff
        # Register array flag here
        if self.logcacheHist:
            self.cacheHist.append('Registered array (Mem map) ' + varName)

    def remove(self, name, index):
        self.cacheManager.cache.pop(name + "_" + str(index), None)
        self.cacheManager.cacheFrequency.pop(name + "_" + str(index), None)
        key_list = list(self.cacheManager.countCache.keys())
        value_list = list(self.cacheManager.countCache.values())
        idx = 0
        flag = False
        for values in value_list:
            idx += 1
            for value in values:
                if (name + "_" + str(index)) == value:
                    flag = True
                    break
            if flag:
                break
        try:
            if flag:
                self.cacheManager.countCache[key_list[idx - 1]].remove(name + "_" + str(index))
                if not self.cacheManager.countCache[key_list[idx - 1]]:
                    self.cacheManager.countCache.pop(key_list[idx - 1], None)
        except:
            print(self.cacheManager.cache)
            print(self.cacheManager.countCache)
            print(self.cacheManager.countCache[key_list[idx - 1]])
            print(key_list)
            print(key_list[idx - 1])
            print(self.cacheManager.countCache[key_list[idx - 1]])
            print(name + "_" + str(index))
            print(self.mem_map)
            os._exit(1)

        self.mem_map.pop(name, None)
        self.num_active_registered_var -= 1

        # Add log to deregister
        if self.logcacheHist:
            self.cacheHist.append('Deregistered array (Mem map) ' + name)

    def compress(
        self, data, varName, index, errBoundMode="PW_REL", compType="blosc", errBound=None, losslesscompressor="zstd"
    ):
        # print("Cprss: ", varName, index)
        # print("Array: ", data)
        # print("Error bound: ",errBound)
        if self.trackBaseline:
            start_time = time()
            if errBound is not None:
                compressor = libpressio.PressioCompressor.from_config(
                    self.generate_compressor_config(compType, errBoundMode, errBound, losslesscompressor)
                )

                self.mem_map[varName][0] = compressor
            else:
                compressor = self.mem_map[varName][0]

            # Compress data
            self.mem_map[varName][1][index] = compressor.encode(data)
            end_time = time()

            # Log array if logging is on
            if self.logcacheHist:
                self.cacheHist.append('Added array: ' + varName)
                self.cacheHist.append('Compressed array (Store) ' + varName)

            # Log compression time and compression calls metrics
            time_diff = end_time - start_time
            self.compression_time_nocache += time_diff
            self.num_compression_calls += 1

        elif self.cacheName(varName, index) not in self.cacheManager.cache:
            self.cacheManager.cache_misses += 1
            start_time = time()
            if errBound is not None:
                compressor = libpressio.PressioCompressor.from_config(
                    self.generate_compressor_config(compType, errBoundMode, errBound, losslesscompressor)
                )

                self.mem_map[varName][0] = compressor
            else:
                compressor = self.mem_map[varName][0]
            # Compress data
            self.mem_map[varName][1][index] = compressor.encode(data)

            # If cache is full, get array name for array which will be evicted and update mem_map (write back cache)
            start_time3 = time()
            if len(self.cacheManager.cache) + 1 > self.cacheManager.cacheSize:
                minKey = min(self.cacheManager.countCache.keys())
                evictArray = self.cacheManager.countCache[minKey][0]
                self.mem_map[evictArray.split('_')[0]][1][index] = compressor.encode(
                    self.cacheManager.cache[evictArray]
                )
                # Add logging for array evicted if logging cache history is on
                if self.logcacheHist:
                    self.cacheHist.append('Evicted array ' + evictArray.split('_')[0])
            end_time3 = time()
            self.compression_time_eviction += end_time3 - start_time3

            # Add to cache
            start_time2 = time()
            self.cacheManager.put(self.cacheName(varName, index), data)
            end_time2 = time()
            self.compression_time_put_only += end_time2 - start_time2

            end_time = time()

            # Add logging for the array to be added and then in compressed state if logging cache history is on
            if self.logcacheHist:
                self.cacheHist.append('Added array: ' + varName)
                self.cacheHist.append('Compressed array (Store) ' + varName)

            # Log compression time and compression calls metrics
            time_diff = end_time - start_time
            self.compression_time += time_diff
            self.num_compression_calls += 1

        else:
            self.cacheManager.cache_hits += 1
            start_time = time()
            combineName = self.cacheName(varName, index)
            self.cacheManager.cache[combineName] = data  # Dictionary access
            prev_count = self.cacheManager.cacheFrequency[combineName]
            self.cacheManager.cacheFrequency[combineName] += 1  # Dictionary access
            count = self.cacheManager.cacheFrequency[combineName]
            self.cacheManager.countCache[prev_count].remove(combineName)
            if not self.cacheManager.countCache[prev_count]:
                self.cacheManager.countCache.pop(prev_count, None)
            if count not in self.cacheManager.countCache.keys():
                self.cacheManager.countCache[count] = []
                self.cacheManager.countCache[count].append(combineName)
            else:
                self.cacheManager.countCache[count].append(combineName)
            end_time = time()

            # Log array if logging is on
            if self.logcacheHist:
                self.cacheHist.append('Updated array ' + varName)

            # Log compression time and compression calls metrics
            time_diff = end_time - start_time
            self.compression_time_update += time_diff

    def cacheName(self, varName, index):
        return varName + "_" + str(index)

    def decompress(self, varName, index, compType=0):
        # print("Decprss: ", varName, index)

        combineName = self.cacheName(varName, index)

        # Decompress array from main memory and return it if no cache mechanism
        if self.trackBaseline:
            start_time = time()
            compressor = self.mem_map[varName][0]
            comp_data = self.mem_map[varName][1][index]
            decomp_data = np.zeros(self.mem_map[varName][2])
            # Get decompressed values of the compressed array
            tmp = compressor.decode(comp_data, decomp_data)
            end_time = time()
            # Log decompression time and decompression calls metrics
            time_diff = end_time - start_time
            self.decompression_time_nocache += time_diff
            self.num_decompression_calls += 1
            # Log cache history if enabled
            if self.logcacheHist:
                self.cacheHist.append('Decompressed array (Load) ' + varName)
            return tmp

        if combineName not in self.cacheManager.cache:
            self.cacheManager.cache_misses += 1
            start_time = time()
            compressor = self.mem_map[varName][0]
            comp_data = self.mem_map[varName][1][index]
            decomp_data = np.zeros(self.mem_map[varName][2])

            # Get decompressed values of the compressed array
            tmp = compressor.decode(comp_data, decomp_data)

            # If cache was full, get array name for array which was evicted and update mem_map (write back cache)
            start_time3 = time()
            if len(self.cacheManager.cache) + 1 > self.cacheManager.cacheSize:
                try:
                    minKey = min(self.cacheManager.countCache.keys())
                    evictArray = self.cacheManager.countCache[minKey][0]
                    self.mem_map[evictArray.split('_')[0]][1][index] = compressor.encode(
                        self.cacheManager.cache[evictArray]
                    )
                    # Add logging for array evicted if logging cache history is on
                    if self.logcacheHist:
                        self.cacheHist.append('Evicted array ' + evictArray.split('_')[0])
                except:
                    print(self.cacheManager.cacheSize)
                    print(len(self.cacheManager.cache))
                    print(self.cacheManager.cache)
                    print(self.cacheManager.cacheFrequency)
                    print(self.cacheManager.countCache)
                    os._exit(1)
            end_time3 = time()
            self.decompression_time_eviction += end_time3 - start_time3

            # Add the array to the cache
            start_time2 = time()
            self.cacheManager.put(combineName, tmp)
            end_time2 = time()
            self.decompression_time_put_only += end_time2 - start_time2

            end_time = time()

            # Log history that array is added if logging cache history is on
            if self.logcacheHist:
                self.cacheHist.append('Added array to cache: ' + varName)
                self.cacheHist.append('Decompressed array (Load) ' + varName)

            # Log decompression time and decompression calls metrics
            time_diff = end_time - start_time
            self.decompression_time += time_diff
            self.num_decompression_calls += 1
            return tmp

            # tmp_mesh = mesh(self.mem_map[varName][2])
            # tmp_mesh[:] = tmp

        else:
            # print ("Found in Cache")
            self.cacheManager.cache_hits += 1
            start_time = time()
            decomp_array = self.cacheManager.get(combineName)
            end_time = time()

            # Log decompression time and decompression calls metrics
            time_diff = end_time - start_time
            self.decompression_time_get += time_diff

            # Log cache history if enabled
            if self.logcacheHist:
                self.cacheHist.append('Decompressed array (Load) ' + varName)
            return decomp_array

    def set_global_compressor_config(self, compType=None, errBoundMode=None, errBound=None, losslesscompressor=None):
        self.compType = self.compType if compType is None else compType
        self.errBoundMode = self.errBoundMode if errBoundMode is None else errBoundMode
        self.errBound = self.errBound if errBound is None else errBound
        self.losslesscompressor = self.losslesscompressor if losslesscompressor is None else losslesscompressor
        for k in self.mem_map:
            self.mem_map[k][0] = libpressio.PressioCompressor.from_config(
                self.generate_compressor_config(
                    self.compType, self.errBoundMode, self.errBound, self.losslesscompressor
                )
            )

    def generate_compressor_config(self, compType=None, errBoundMode=None, errBound=None, losslesscompressor=None):
        compType = self.compType if compType is None else compType
        errBoundMode = self.errBoundMode if errBoundMode is None else errBoundMode
        errBound = self.errBound if errBound is None else errBound
        losslesscompressor = self.losslesscompressor if losslesscompressor is None else losslesscompressor
        # print('Error bound: ',errBound)
        # return {
        #     # configure which compressor to use
        #     "compressor_id": compType,
        #     # configure the set of metrics to be gathered
        #     "early_config": {
        #         "pressio:metric": "composite",
        #         "composite:plugins": ["time", "size", "error_stat"],
        #     },
        #     # configure SZ
        #     "compressor_config": {
        #         "pressio:abs": errBound,
        #     },
        # }
        return {
            # configure which compressor to use
            "compressor_id": "blosc",
            # configure the set of metrics to be gathered
            "early_config": {
                "pressio:metric": "composite",
                "composite:plugins": ["time", "size", "error_stat"],
            },
            # configure SZ
            "compressor_config": {
                "blosc:compressor": "zstd",
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

    def getStats(self, varName, index=0):
        compressor = self.mem_map[varName][0]
        return compressor.get_metrics()

    # Calculate compression ratio for all arrays stored in main memory and cache
    def sumCompressionRatio(self):
        keys_memmap = self.mem_map.keys()
        if not self.trackBaseline:
            list_keys = self.cacheManager.cache.keys()
        sumTotalSize = 0
        sumCompSize = 0
        sumCacheDecompSize = 0
        # sumCR = []

        # Get total size of all arrays in main memory and size of all compressed arrays in main memory
        for k in keys_memmap:
            sumTotalSize += self.getStats(k, 0)['size:uncompressed_size']
            sumCompSize += self.getStats(k, 0)['size:compressed_size']

        # Get total size of all arrays in cache
        if not self.trackBaseline:
            for key in list_keys:
                key = key.split('_')[0]
                sumCacheDecompSize += self.getStats(key, 0)['size:uncompressed_size']
        return sumTotalSize / (sumCompSize + sumCacheDecompSize)

    def getWrapperTime(self):
        return self.registerVars_time + self.compression_time + self.decompression_time


if __name__ == "__main__":
    arr = np.random.rand(100, 100)
    # declare global instance of memory
    memory = CRAM_Manager(errBoundMode="PW_REL", compType="sz3", errBound=1e-5)

    memory.registerVar(
        "cat",
        arr.shape,
        dtype=np.dtype("float64"),
        numVectors=1,
        errBoundMode="PW_REL",
        compType="sz3",
        errBound=0.1,
    )
    memory.compress(arr, "cat", 0)
    result = memory.decompress("cat", 0)
    print(arr)
    print("\n")
    print(result)
    print("\n")
    error = arr - result
    print(error)
    print("\n")
    print("Cache History\n")
    print(memory.cacheHist)
    print("\nCache\n")
    print(memory.cacheManager.cache)
    print("Cache History\n")
    print(memory.cacheHist)
    print("\nCache\n")
    print(memory.cacheManager.cache)
    print(memory.sumCompressionRatio())

# implement cache replacement policy
# LFU - use a counter for that array and then based on that counter when it comes to evict, choose the minimum counter and evict it
# Also write back cache
