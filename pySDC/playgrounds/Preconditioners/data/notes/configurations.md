# pySDC configurations for optimizing preconditioners
We record the output of the controller at the beginning of a single run here for the sake of sustainability.

For advection:
```
Controller: <class 'pySDC.implementations.controller_classes.controller_nonMPI.controller_nonMPI'>                                     
--> mssdc_jac = False                                                                                                                  
    predict_type = None                                                                                                                
    all_to_done = False                                                                                                                
--> logger_level = 15                                                                                                                  
    log_to_file = False                                                                                                                
    dump_setup = True                                                                                                                  
    fname = run_pid36756.log                                                                                                           
    use_iteration_estimator = False                                                                                                    
--> hook_class = <class 'pySDC.playgrounds.Preconditioners.hooks.log_cost'>                                                            
                                                                                                                                       
Step: <class 'pySDC.core.Step.step'>                                                                                                   
--> maxiter = 5                                                                                                                        
    Level: <class 'pySDC.core.Level.level'>                                                                                            
        Level  0                                                                                                                       
-->         dt = 0.05                                                                                                                  
            dt_initial = 0.05                                                                                                          
            restol = -1.0                                                                                                              
            nsweeps = 1                                                                                                                
            residual_type = full_abs                                                                                                   
-->         Problem: <class 'pySDC.implementations.problem_classes.AdvectionEquation_ND_FD.advectionNd'>                               
-->             freq = (-1,)                                                                                                           
-->             nvars = (512,)                                                                                                         
-->             c = 1.0                                                                                                                
-->             type = backward                                                                                                        
-->             order = 5                                                                                                              
-->             bc = periodic                                                                                                          
-->             sigma = 0.06                                                                                                           
                stencil_type = center                                                                                                  
                lintol = 1e-12                                                                                                         
                liniter = 10000                                                                                                        
                direct_solver = True                                                                                                   
                ndim = 1                                                                                                               
-->             Data type u: <class 'pySDC.implementations.datatype_classes.mesh.mesh'>                                                
-->             Data type f: <class 'pySDC.implementations.datatype_classes.mesh.mesh'>                                                
-->             Sweeper: <class 'pySDC.playgrounds.Preconditioners.diagonal_precon_sweeper.DiagPrecon'>                                
                    do_coll_update = False                                                                                             
-->                 initial_guess = spread                                                                                             
-->                 quad_type = RADAU-RIGHT                                                                                            
-->                 num_nodes = 3                                                                                                      
-->                 QI = IE                                                                                                            
-->                 diagonal_elements = [0.2 0.4 0.4]                                                                                  
-->                 first_row = [0. 0. 0.]                                                                                             
-->                 Collocation: <class 'pySDC.core.Collocation.CollBase'>                          
```

And for diffusion:
```
Controller: <class 'pySDC.implementations.controller_classes.controller_nonMPI.controller_nonMPI'>                                     
--> mssdc_jac = False                                                                                                                  
    predict_type = None                                                                                                                
    all_to_done = False                                                                                                                
--> logger_level = 15                                                                                                                  
    log_to_file = False                                                                                                                
    dump_setup = True                                                                                                                  
    fname = run_pid36779.log                                                                                                           
    use_iteration_estimator = False                                                                                                    
--> hook_class = <class 'pySDC.playgrounds.Preconditioners.hooks.log_cost'>                                                            
                                                                                                                                       
Step: <class 'pySDC.core.Step.step'>                                                                                                   
--> maxiter = 5                                                                                                                        
    Level: <class 'pySDC.core.Level.level'>                                                                                            
        Level  0                                                                                                                       
-->         dt = 0.05                                                                                                                  
            dt_initial = 0.05                                                                                                          
            restol = -1.0                                                                                                              
            nsweeps = 1                                                                                                                
            residual_type = full_abs                                                                                                   
-->         Problem: <class 'pySDC.implementations.problem_classes.HeatEquation_ND_FD.heatNd_unforced'>                                
-->             freq = (-1,)                                                                                                           
-->             nvars = (128,)                                                                                                         
-->             nu = 1.0                                                                                                               
-->             type = forward                                                                                                         
-->             order = 2                                                                                                              
-->             bc = periodic                                                                                                          
-->             direct_solver = True                                                                                                   
-->             lintol = None                                                                                                          
-->             liniter = None                                                                                                         
-->             sigma = 0.06                                                                                                           
                stencil_type = center                               
                ndim = 1                                            
-->             Data type u: <class 'pySDC.implementations.datatype_classes.mesh.mesh'>                                                 
-->             Data type f: <class 'pySDC.implementations.datatype_classes.mesh.mesh'>                                                 
-->             Sweeper: <class 'pySDC.playgrounds.Preconditioners.diagonal_precon_sweeper.DiagPrecon'>                                 
                    do_coll_update = False                          
-->                 initial_guess = spread                          
-->                 quad_type = RADAU-RIGHT                         
-->                 num_nodes = 3                                   
-->                 QI = IE                                         
-->                 diagonal_elements = [0.2 0.4 0.4]               
-->                 first_row = [0. 0. 0.]                          
-->                 Collocation: <class 'pySDC.core.Collocation.CollBase'>                                              
```