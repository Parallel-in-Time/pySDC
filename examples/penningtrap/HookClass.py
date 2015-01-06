from pySDC.Hooks import hooks

import matplotlib.pyplot as plt

class particles_output(hooks):

    def __init__(self):
        super(particles_output,self).__init__()

        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection='3d')
        self.ax.set_xlim3d([-20,20])
        self.ax.set_ylim3d([-20,20])
        self.ax.set_zlim3d([-20,20])
        plt.ion()
        self.sframe = None

    # def dump_iteration(self):
    #     L = self.level
    #
    #     oldcol = self.sframe
    #     self.sframe = self.ax.scatter(L.u[-1].pos.values[0::3],L.u[-1].pos.values[1::3],L.u[-1].pos.values[2::3])
    #     # Remove old line collection before drawing
    #     if oldcol is not None:
    #         self.ax.collections.remove(oldcol)
    #     plt.pause(.001)
    #
    #     return None


    def dump_step(self):
        L = self.level

        print('plotting particles...')

        oldcol = self.sframe
        self.sframe = self.ax.scatter(L.u[-1].pos.values[0::3],L.u[-1].pos.values[1::3],L.u[-1].pos.values[2::3])
        # Remove old line collection before drawing
        if oldcol is not None:
            self.ax.collections.remove(oldcol)
        plt.pause(.001)

        return None
