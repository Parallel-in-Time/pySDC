



def setup_hierarchy(problem_class,dtype_u,dtype_f,collocation_class,num_nodes,sweeper_class,lparams,sparams,cparams):

    assert 'parblocks' in sparams

    NP = sparams['parblocks']

    MS = []
    for p in range(NP):

        L = levclass.level( problem_class       =   problem_class,
                            problem_params      =   cparams,
                            dtype_u             =   dtype_u,
                            dtype_f             =   dtype_f,
                            collocation_class   =   collocation_class,
                            num_nodes           =   num_nodes,
                            sweeper_class       =   sweeper_class,
                            level_params        =   lparams,
                            id                  =   'R'+str(p)+'+L0')

        MS.append()



    return MS