import numpy as np

np.bool = np.bool_
import os


class Cache:
    def __init__(self):
        self.cache = {}
        self.cacheFrequency = {}
        self.countCache = {}
        self.cacheSize = 0
        self.cacheInvalidates = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_accesses = 0

    # Return array block existing in cache, function is called only if varName key exists so array will be returned
    def get(self, varName):
        # get initial count (frequency) of array to update countCache dictionary and remove it
        prev_count = self.cacheFrequency[varName]

        # Increment frequency counter for access to this array
        self.cacheFrequency[varName] += 1
        count = self.cacheFrequency[varName]

        # In countCache dictionary, delete the value varName in key count and add varName to key count's list
        self.countCache[prev_count].remove(varName)
        if not self.countCache[prev_count]:
            self.countCache.pop(prev_count, None)

        if count not in self.countCache.keys():
            self.countCache[count] = []
            self.countCache[count].append(varName)
        else:
            self.countCache[count].append(varName)
        return self.cache[varName]

    # Add array to the cache if it doesn't exist or update the existing block of array
    def put(self, varName, data):
        # Implement LFU cache eviction policy
        if len(self.cache) + 1 > self.cacheSize:
            # get minimum count and then the first in the list is evicted and the new array is added there
            try:
                # print('Here:')
                # print(self.cache)
                # print(self.countCache)
                minKey = min(self.countCache.keys())
                # print(minKey)

                evictArray = self.countCache[minKey][0]
                # print(evictArray)
                self.countCache[minKey].remove(evictArray)
                if not self.countCache[minKey]:
                    self.countCache.pop(minKey, None)
                self.cache.pop(evictArray, None)
                self.cacheFrequency.pop(evictArray, None)
                self.cacheInvalidates += 1
                # print(self.cache)
                # print(self.countCache)
                # print('Array Eviction done')

                # Add the new array
                self.cache[varName] = data
                self.cacheFrequency[varName] = 1
                count = self.cacheFrequency[varName]
                if count not in self.countCache.keys():
                    self.countCache[count] = []
                    self.countCache[count].append(varName)
                else:
                    self.countCache[count].append(varName)

                # print(self.cache)
                # print(self.countCache)
                # print('New Array Added')
            except KeyError:
                print('Failed to evict correctly')
                os._exit(1)
        else:
            self.cache[varName] = data
            self.cacheFrequency[varName] = 1
            count = self.cacheFrequency[varName]
            if count not in self.countCache.keys():
                self.countCache[count] = []
                self.countCache[count].append(varName)
            else:
                self.countCache[count].append(varName)

if __name__ == "__main__":
    arr = np.random.rand(100, 100)
    # declare global instance of memory
    memory = Cache()
    print("Cache instance:")