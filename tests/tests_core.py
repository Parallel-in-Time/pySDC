import numpy as np


#def test_python_version():
#    import sys
#    assert sys.version_info >= (3, 3), "Need Python 3.3 at least"

# def test_return_types():
    # import pySDC.problem as pr
    # import CollocationClasses as collclass
    # import sweeper as sw
    #
    # # coll = collclass.CollGaussLobatto(5,0,1)
    # prob = pr.forced_heat_1d(15,0.1)
    #
    # u = prob.u_exact(1)
    # assert type(u) == list


def test_errors():
    classes = ['DataError']
    for subclass in classes:
        yield check_error, subclass

def check_error(subclass):
    import pySDC.Errors

    err = getattr(pySDC.Errors, subclass)

    try:
        raise err('bla')
        assert False
    except err:
        assert True




def test_datatypes_mesh():

    init = [10,(10,10),(10,10,10)]
    for i in init:
        yield check_datatypes_mesh, i


def check_datatypes_mesh(init):
    import pySDC.implementations.datatype_classes.mesh as m


    m1 = m.mesh(init)
    m2 = m.mesh(m1)


    m1.values[:] = 1.0
    m2.values[:] = 2.0

    m3 = m1 + m2
    m4 = m1 - m2
    m5 = 0.1*m1
    m6 = m1

    m7 = abs(m1)

    m8 = m.mesh(m1)

    assert isinstance(m3,type(m1))
    assert isinstance(m4,type(m1))
    assert isinstance(m5,type(m1))
    assert isinstance(m6,type(m1))
    assert isinstance(m7,float)

    assert m2 is not m1
    assert m3 is not m1
    assert m4 is not m1
    assert m5 is not m1
    assert m6 is m1

    assert np.shape(m3.values) == np.shape(m1.values)
    assert np.shape(m4.values) == np.shape(m1.values)
    assert np.shape(m5.values) == np.shape(m1.values)

    assert np.all(m1.values==1.0)
    assert np.all(m2.values==2.0)
    assert np.all(m3.values==3.0)
    assert np.all(m4.values==-1.0)
    assert np.all(m5.values==0.1)
    assert np.all(m8.values==1.0)
    assert m7 >= 0

def test_datatypes_particles():

    init = [1,10]
    for i in init:
        yield check_datatypes_particles, i


def check_datatypes_particles(init):
    from pySDC.implementations.datatype_classes.particles import particles
    from pySDC.implementations.datatype_classes.particles import acceleration


    p1 = particles(init)
    p2 = particles(p1)
    p5 = particles(init)

    p1.pos.values[:] = 1.0
    p2.pos.values[:] = 2.0
    p1.vel.values[:] = 10.0
    p2.vel.values[:] = 20.0

    p3 = p1 + p2
    p4 = p1 - p2

    p5.pos = 0.1*p1.vel
    p6 = p1

    p7 = abs(p1)

    a1 = acceleration(init)
    a2 = acceleration(a1)
    p8 = particles(p1)

    a1.values[:] = 100.0
    a2.values[:] = 200.0

    a3 = a1 + a2

    p8.vel = 0.1*a1
    p8.pos = 0.1*(0.1*a1)

    assert isinstance(p3,type(p1))
    assert isinstance(p4,type(p1))
    assert isinstance(p5.pos,type(p1.pos))
    assert isinstance(p6,type(p1))
    assert isinstance(p7,float)
    assert isinstance(a2,type(a1))
    assert isinstance(p8.pos,type(p1.pos))
    assert isinstance(p8.vel,type(p1.vel))
    assert isinstance(0.1*0.1*a1,type(p1.vel))

    assert p2 is not p1
    assert p3 is not p1
    assert p4 is not p1
    assert p5 is not p1
    assert p6 is p1
    assert a2 is not a1
    assert a3 is not a1

    assert np.shape(p3.pos.values) == np.shape(p1.pos.values)
    assert np.shape(p4.pos.values) == np.shape(p1.pos.values)
    assert np.shape(p3.vel.values) == np.shape(p1.vel.values)
    assert np.shape(p4.vel.values) == np.shape(p1.vel.values)
    assert np.shape(a2.values) == np.shape(a1.values)

    assert np.all(p3.pos.values==3.0)
    assert np.all(p4.pos.values==-1.0)
    assert np.all(p3.vel.values==30.0)
    assert np.all(p4.vel.values==-10.0)
    assert np.all(p5.pos.values==1.0)
    assert p7 >= 0
    assert np.all(p8.pos.values==1.0)
    assert np.all(p8.vel.values==10.0)
    assert np.all(a3.values==300.0)
