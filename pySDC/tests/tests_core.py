import numpy as np



def test_errors():
    classes = ['DataError']
    for subclass in classes:
        yield check_error, subclass

def check_error(subclass):
    import pySDC.core.Errors

    err = getattr(pySDC.core.Errors, subclass)

    try:
        raise err('bla')
    except err:
        assert True




def test_datatypes_mesh():

    init = [10,(10,10),(10,10,10)]
    for i in init:
        yield check_datatypes_mesh, i


def check_datatypes_mesh(init):
    import pySDC.implementations.datatype_classes.parallel_mesh as m


    m1 = m.parallel_mesh(init)
    m2 = m.parallel_mesh(m1)


    m1[:] = 1.0
    m2[:] = 2.0

    m3 = m1 + m2
    m4 = m1 - m2
    m5 = 0.1*m1
    m6 = m1

    m7 = abs(m1)

    m8 = m.parallel_mesh(m1)

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

    assert np.shape(m3) == np.shape(m1)
    assert np.shape(m4) == np.shape(m1)
    assert np.shape(m5) == np.shape(m1)

    assert np.all(m1==1.0)
    assert np.all(m2==2.0)
    assert np.all(m3==3.0)
    assert np.all(m4==-1.0)
    assert np.all(m5==0.1)
    assert np.all(m8==1.0)
    assert m7 >= 0

def test_datatypes_particles():

    init = [1,10]
    for i in init:
        yield check_datatypes_particles, i


def check_datatypes_particles(init):
    from pySDC.implementations.datatype_classes.particles import particles
    from pySDC.implementations.datatype_classes.particles import acceleration


    p1 = particles((init, None, np.dtype('float64')))
    p2 = particles(p1)
    p5 = particles((init, None, np.dtype('float64')))

    p1.pos[:] = 1.0
    p2.pos[:] = 2.0
    p1.vel[:] = 10.0
    p2.vel[:] = 20.0

    p3 = p1 + p2
    p4 = p1 - p2

    p5.pos[:] = 0.1*p1.vel
    p6 = p1

    p7 = abs(p1)

    a1 = acceleration((init, None, np.dtype('float64')))
    a2 = acceleration(a1)
    p8 = particles(p1)

    a1[:] = 100.0
    a2[:] = 200.0

    a3 = a1 + a2

    p8.vel[:] = 0.1*a1
    p8.pos[:] = 0.1*(0.1*a1)

    assert isinstance(p3,type(p1))
    assert isinstance(p4,type(p1))
    assert isinstance(p5.pos,type(p1.pos))
    assert isinstance(p6,type(p1))
    assert isinstance(p7,float)
    assert isinstance(a2,type(a1))
    assert isinstance(p8.pos,type(p1.pos))
    assert isinstance(p8.vel,type(p1.vel))

    assert p2 is not p1
    assert p3 is not p1
    assert p4 is not p1
    assert p5 is not p1
    assert p6 is p1
    assert a2 is not a1
    assert a3 is not a1

    assert np.shape(p3.pos) == np.shape(p1.pos)
    assert np.shape(p4.pos) == np.shape(p1.pos)
    assert np.shape(p3.vel) == np.shape(p1.vel)
    assert np.shape(p4.vel) == np.shape(p1.vel)
    assert np.shape(a2) == np.shape(a1)

    assert np.all(p3.pos==3.0)
    assert np.all(p4.pos==-1.0)
    assert np.all(p3.vel==30.0)
    assert np.all(p4.vel==-10.0)
    assert np.all(p5.pos==1.0)
    assert p7 >= 0
    assert np.all(p8.pos==1.0)
    assert np.all(p8.vel==10.0)
    assert np.all(a3==300.0)
