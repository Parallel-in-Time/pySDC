# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/Parallel-in-Time/pySDC/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                                                |    Stmts |     Miss |   Cover |   Missing |
|---------------------------------------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| pySDC/core/base\_transfer.py                                                                        |      107 |        1 |     99% |        12 |
| pySDC/core/check\_convergence.py                                                                    |       70 |        2 |     97% |   84, 170 |
| pySDC/core/collocation.py                                                                           |       42 |        1 |     98% |        78 |
| pySDC/core/common.py                                                                                |       23 |        0 |    100% |           |
| pySDC/core/controller.py                                                                            |      187 |        1 |     99% |       104 |
| pySDC/core/convergence\_controller.py                                                               |      111 |       18 |     84% |6-7, 438-448, 461-468, 489, 572, 585-589 |
| pySDC/core/default\_hook.py                                                                         |       19 |        0 |    100% |           |
| pySDC/core/errors.py                                                                                |        9 |        0 |    100% |           |
| pySDC/core/hooks.py                                                                                 |       57 |        2 |     96% |     6, 87 |
| pySDC/core/level.py                                                                                 |       71 |        0 |    100% |           |
| pySDC/core/problem.py                                                                               |       45 |        2 |     96% |   42, 166 |
| pySDC/core/space\_transfer.py                                                                       |       20 |        0 |    100% |           |
| pySDC/core/step.py                                                                                  |      118 |        4 |     97% |113-114, 142-143 |
| pySDC/core/sweeper.py                                                                               |      131 |        8 |     94% |11, 69-70, 298-301, 321-322 |
| pySDC/core/timings.py                                                                               |       96 |        0 |    100% |           |
| pySDC/helpers/NCCL\_communicator.py                                                                 |       98 |       10 |     90% |200-205, 211-216 |
| pySDC/helpers/ParaDiagHelper.py                                                                     |       36 |        0 |    100% |           |
| pySDC/helpers/blocks.py                                                                             |       68 |        9 |     87% |47-49, 52-55, 70-71 |
| pySDC/helpers/fft\_helper.py                                                                        |        6 |        1 |     83% |        26 |
| pySDC/helpers/fieldsIO.py                                                                           |      308 |        7 |     98% |89-90, 98-99, 418, 606-607 |
| pySDC/helpers/firedrake\_ensemble\_communicator.py                                                  |       53 |        8 |     85% |71, 81-85, 90, 100, 132 |
| pySDC/helpers/plot\_helper.py                                                                       |       33 |        2 |     94% |  118, 167 |
| pySDC/helpers/problem\_helper.py                                                                    |      103 |        0 |    100% |           |
| pySDC/helpers/pySDC\_as\_gusto\_time\_discretization.py                                             |       92 |        2 |     98% |  115, 157 |
| pySDC/helpers/pysdc\_helper.py                                                                      |       19 |        1 |     95% |        89 |
| pySDC/helpers/setup\_helper.py                                                                      |       19 |        0 |    100% |           |
| pySDC/helpers/spectral\_helper.py                                                                   |      796 |       57 |     93% |155, 196, 250, 353-354, 568, 924, 933, 1019-1022, 1087, 1175, 1274, 1323, 1624-1670, 1763-1766, 1800, 1811-1833, 1949, 2027-2029 |
| pySDC/helpers/stats\_helper.py                                                                      |       36 |        0 |    100% |           |
| pySDC/helpers/testing.py                                                                            |       30 |        0 |    100% |           |
| pySDC/helpers/transfer\_helper.py                                                                   |      142 |        0 |    100% |           |
| pySDC/helpers/visualization\_tools.py                                                               |       40 |        0 |    100% |           |
| pySDC/helpers/vtkIO.py                                                                              |       51 |        1 |     98% |        90 |
| pySDC/implementations/controller\_classes/ParaDiag.py                                               |       55 |        3 |     95% |50, 57, 116 |
| pySDC/implementations/controller\_classes/controller\_MPI.py                                        |      299 |       46 |     85% |72, 238, 262, 284, 436, 446, 454, 458, 463, 484, 517, 550, 554, 574, 579, 603-623, 636, 649, 669-685 |
| pySDC/implementations/controller\_classes/controller\_ParaDiag\_MPI.py                              |       91 |        0 |    100% |           |
| pySDC/implementations/controller\_classes/controller\_ParaDiag\_nonMPI.py                           |       86 |        0 |    100% |           |
| pySDC/implementations/controller\_classes/controller\_nonMPI.py                                     |      297 |        3 |     99% |415-416, 458 |
| pySDC/implementations/convergence\_controller\_classes/adaptive\_alpha.py                           |       36 |        0 |    100% |           |
| pySDC/implementations/convergence\_controller\_classes/adaptive\_collocation.py                     |       77 |        1 |     99% |       249 |
| pySDC/implementations/convergence\_controller\_classes/adaptivity.py                                |      257 |       19 |     93% |130-131, 265, 274, 276, 299-308, 400, 407, 584, 729, 745, 1001-1002 |
| pySDC/implementations/convergence\_controller\_classes/basic\_restarting.py                         |       99 |        1 |     99% |       177 |
| pySDC/implementations/convergence\_controller\_classes/check\_iteration\_estimator.py               |       46 |        1 |     98% |        39 |
| pySDC/implementations/convergence\_controller\_classes/crash.py                                     |       45 |        0 |    100% |           |
| pySDC/implementations/convergence\_controller\_classes/estimate\_contraction\_factor.py             |       23 |        0 |    100% |           |
| pySDC/implementations/convergence\_controller\_classes/estimate\_embedded\_error.py                 |      113 |       25 |     78% |34, 37-38, 58, 101, 110-117, 221, 251-252, 266-287 |
| pySDC/implementations/convergence\_controller\_classes/estimate\_extrapolation\_error.py            |      161 |        5 |     97% |103, 110, 364, 395, 399 |
| pySDC/implementations/convergence\_controller\_classes/estimate\_polynomial\_error.py               |       97 |       10 |     90% |55, 210, 236-245 |
| pySDC/implementations/convergence\_controller\_classes/hotrod.py                                    |       38 |        5 |     87% |85, 89, 96, 126-127 |
| pySDC/implementations/convergence\_controller\_classes/inexactness.py                               |       24 |        2 |     92% |     50-54 |
| pySDC/implementations/convergence\_controller\_classes/interpolate\_between\_restarts.py            |       41 |        3 |     93% | 55, 93-94 |
| pySDC/implementations/convergence\_controller\_classes/spread\_step\_sizes.py                       |       61 |        0 |    100% |           |
| pySDC/implementations/convergence\_controller\_classes/step\_size\_limiter.py                       |       63 |        0 |    100% |           |
| pySDC/implementations/convergence\_controller\_classes/store\_uold.py                               |       14 |        0 |    100% |           |
| pySDC/implementations/datatype\_classes/container.py                                                |       25 |        0 |    100% |           |
| pySDC/implementations/datatype\_classes/cupy\_mesh.py                                               |       65 |        4 |     94% |5-6, 12-13 |
| pySDC/implementations/datatype\_classes/fenics\_mesh.py                                             |       33 |        0 |    100% |           |
| pySDC/implementations/datatype\_classes/firedrake\_mesh.py                                          |       49 |        1 |     98% |        45 |
| pySDC/implementations/datatype\_classes/mesh.py                                                     |       72 |        0 |    100% |           |
| pySDC/implementations/datatype\_classes/particles.py                                                |       82 |       20 |     76% |63-73, 173-174, 188, 202-207 |
| pySDC/implementations/datatype\_classes/petsc\_vec.py                                               |       30 |        0 |    100% |           |
| pySDC/implementations/hooks/AllenCahn\_monitor.py                                                   |       60 |        0 |    100% |           |
| pySDC/implementations/hooks/log\_GPU\_timings.py                                                    |       11 |        0 |    100% |           |
| pySDC/implementations/hooks/log\_embedded\_error\_estimate.py                                       |       20 |        0 |    100% |           |
| pySDC/implementations/hooks/log\_errors.py                                                          |       50 |        0 |    100% |           |
| pySDC/implementations/hooks/log\_extrapolated\_error\_estimate.py                                   |        6 |        0 |    100% |           |
| pySDC/implementations/hooks/log\_restarts.py                                                        |        6 |        0 |    100% |           |
| pySDC/implementations/hooks/log\_solution.py                                                        |      144 |        6 |     96% |173, 286-287, 348, 388, 417 |
| pySDC/implementations/hooks/log\_step\_size.py                                                      |        6 |        0 |    100% |           |
| pySDC/implementations/hooks/log\_work.py                                                            |       18 |        0 |    100% |           |
| pySDC/implementations/hooks/plotting.py                                                             |        2 |        0 |    100% |           |
| pySDC/implementations/problem\_classes/AcousticAdvection\_1D\_FD\_imex.py                           |       52 |        0 |    100% |           |
| pySDC/implementations/problem\_classes/AdvectionDiffusionEquation\_1D\_FFT.py                       |       63 |        0 |    100% |           |
| pySDC/implementations/problem\_classes/AdvectionEquation\_ND\_FD.py                                 |       23 |        8 |     65% |   115-125 |
| pySDC/implementations/problem\_classes/AllenCahn\_1D\_FD.py                                         |      225 |       10 |     96% |   474-485 |
| pySDC/implementations/problem\_classes/AllenCahn\_2D\_FD.py                                         |      212 |        8 |     96% |243, 316-319, 441-445 |
| pySDC/implementations/problem\_classes/AllenCahn\_2D\_FFT.py                                        |       85 |       10 |     88% |106, 232-245 |
| pySDC/implementations/problem\_classes/AllenCahn\_MPIFFT.py                                         |       92 |        8 |     91% |167, 229, 236, 243, 249, 264, 271, 277 |
| pySDC/implementations/problem\_classes/AllenCahn\_Temp\_MPIFFT.py                                   |      126 |       22 |     83% |279-303, 319-321 |
| pySDC/implementations/problem\_classes/Auzinger\_implicit.py                                        |       38 |        0 |    100% |           |
| pySDC/implementations/problem\_classes/Battery.py                                                   |      165 |        2 |     99% |     82-83 |
| pySDC/implementations/problem\_classes/Boussinesq\_2D\_FD\_imex.py                                  |       67 |        1 |     99% |       108 |
| pySDC/implementations/problem\_classes/Brusselator.py                                               |       44 |        6 |     86% |57-58, 145-149 |
| pySDC/implementations/problem\_classes/BuckConverter.py                                             |       47 |        0 |    100% |           |
| pySDC/implementations/problem\_classes/Burgers.py                                                   |       96 |        5 |     95% |   371-377 |
| pySDC/implementations/problem\_classes/DiscontinuousTestODE.py                                      |       87 |        0 |    100% |           |
| pySDC/implementations/problem\_classes/FastWaveSlowWave\_0D.py                                      |       40 |        0 |    100% |           |
| pySDC/implementations/problem\_classes/FermiPastaUlamTsingou.py                                     |       40 |        1 |     98% |        61 |
| pySDC/implementations/problem\_classes/FullSolarSystem.py                                           |       43 |        0 |    100% |           |
| pySDC/implementations/problem\_classes/GeneralizedFisher\_1D\_FD\_implicit.py                       |       57 |        2 |     96% |  178, 181 |
| pySDC/implementations/problem\_classes/GeneralizedFisher\_1D\_PETSc.py                              |      243 |        2 |     99% |  126, 223 |
| pySDC/implementations/problem\_classes/GenericGusto.py                                              |      110 |        8 |     93% |91, 156, 182-186, 307-311 |
| pySDC/implementations/problem\_classes/GrayScott\_1D\_FEniCS\_implicit.py                           |       87 |        2 |     98% |  122, 126 |
| pySDC/implementations/problem\_classes/GrayScott\_2D\_PETSc\_periodic.py                            |      305 |        2 |     99% |  175, 281 |
| pySDC/implementations/problem\_classes/GrayScott\_MPIFFT.py                                         |      297 |        6 |     98% |266-277, 290-291 |
| pySDC/implementations/problem\_classes/HarmonicOscillator.py                                        |       54 |       17 |     69% |71-78, 102-111, 114-121 |
| pySDC/implementations/problem\_classes/HeatEquation\_1D\_FEniCS\_matrix\_forced.py                  |      123 |        9 |     93% |191-193, 228-233 |
| pySDC/implementations/problem\_classes/HeatEquation\_2D\_PETSc\_forced.py                           |       90 |        0 |    100% |           |
| pySDC/implementations/problem\_classes/HeatEquation\_Chebychev.py                                   |      220 |        0 |    100% |           |
| pySDC/implementations/problem\_classes/HeatEquation\_ND\_FD.py                                      |       66 |        8 |     88% |88, 107, 118-119, 237-238, 273-274 |
| pySDC/implementations/problem\_classes/HeatFiredrake.py                                             |       63 |        1 |     98% |       169 |
| pySDC/implementations/problem\_classes/HenonHeiles.py                                               |       31 |        0 |    100% |           |
| pySDC/implementations/problem\_classes/LogisticEquation.py                                          |       40 |       20 |     50% |   119-155 |
| pySDC/implementations/problem\_classes/Lorenz.py                                                    |       54 |        3 |     94% |155, 158-159 |
| pySDC/implementations/problem\_classes/NonlinearSchroedinger\_MPIFFT.py                             |       60 |        4 |     93% |80, 89, 229-230 |
| pySDC/implementations/problem\_classes/OuterSolarSystem.py                                          |       58 |        0 |    100% |           |
| pySDC/implementations/problem\_classes/PenningTrap\_3D.py                                           |      124 |       13 |     90% |5-8, 124-138, 183 |
| pySDC/implementations/problem\_classes/Piline.py                                                    |       44 |        0 |    100% |           |
| pySDC/implementations/problem\_classes/Quench.py                                                    |      151 |       34 |     77% |205-206, 267-269, 340, 411-461 |
| pySDC/implementations/problem\_classes/RayleighBenard3D.py                                          |      223 |        4 |     98% |107, 356, 455, 514 |
| pySDC/implementations/problem\_classes/RayleighBenard.py                                            |      255 |       15 |     94% |99, 325, 345, 492, 548, 568, 734-749 |
| pySDC/implementations/problem\_classes/TestEquation\_0D.py                                          |      101 |        0 |    100% |           |
| pySDC/implementations/problem\_classes/Van\_der\_Pol\_implicit.py                                   |       64 |        1 |     98% |       182 |
| pySDC/implementations/problem\_classes/VorticityVelocity\_2D\_FEniCS\_periodic.py                   |      108 |       51 |     53% |164-169, 188-197, 216-221, 240-243, 260-263, 280-282, 298-324, 407-412, 431-441, 460-464, 486-489 |
| pySDC/implementations/problem\_classes/acoustic\_helpers/buildWave1DMatrix.py                       |       24 |        0 |    100% |           |
| pySDC/implementations/problem\_classes/acoustic\_helpers/standard\_integrators.py                   |      259 |       11 |     96% |   301-313 |
| pySDC/implementations/problem\_classes/boussinesq\_helpers/build2DFDMatrix.py                       |       39 |        6 |     85% |40-42, 45-47 |
| pySDC/implementations/problem\_classes/boussinesq\_helpers/buildBoussinesq2DMatrix.py               |       25 |        0 |    100% |           |
| pySDC/implementations/problem\_classes/boussinesq\_helpers/buildFDMatrix.py                         |       42 |        0 |    100% |           |
| pySDC/implementations/problem\_classes/boussinesq\_helpers/helper\_classes.py                       |       19 |        1 |     95% |        14 |
| pySDC/implementations/problem\_classes/boussinesq\_helpers/standard\_integrators.py                 |      394 |      112 |     72% |21-25, 28-32, 36-50, 266-271, 274-276, 282, 288-301, 309-313, 316, 319-320, 326, 332-345, 372-378, 459-466, 472, 492-496, 499-511, 514-526, 551-580 |
| pySDC/implementations/problem\_classes/boussinesq\_helpers/unflatten.py                             |        7 |        0 |    100% |           |
| pySDC/implementations/problem\_classes/generic\_MPIFFT\_Laplacian.py                                |       89 |        0 |    100% |           |
| pySDC/implementations/problem\_classes/generic\_ND\_FD.py                                           |       96 |        2 |     98% |   156-157 |
| pySDC/implementations/problem\_classes/generic\_spectral.py                                         |      244 |       42 |     83% |117, 271-273, 340-342, 357-358, 395, 411, 454, 476, 479, 514, 521-526, 546-583 |
| pySDC/implementations/problem\_classes/nonlinear\_ODE\_1.py                                         |       41 |        0 |    100% |           |
| pySDC/implementations/problem\_classes/odeScalar.py                                                 |       53 |        0 |    100% |           |
| pySDC/implementations/problem\_classes/odeSystem.py                                                 |      187 |        6 |     97% |602-615, 951-964 |
| pySDC/implementations/problem\_classes/polynomial\_test\_problem.py                                 |       36 |        0 |    100% |           |
| pySDC/implementations/sweeper\_classes/Multistep.py                                                 |       91 |        6 |     93% |48-52, 123-125 |
| pySDC/implementations/sweeper\_classes/ParaDiagSweepers.py                                          |       71 |        8 |     89% |38, 130, 155-165 |
| pySDC/implementations/sweeper\_classes/Runge\_Kutta.py                                              |      361 |       25 |     93% |235-237, 250-262, 444-458, 515-517, 521-523 |
| pySDC/implementations/sweeper\_classes/Runge\_Kutta\_Nystrom.py                                     |       98 |        4 |     96% | 52-54, 68 |
| pySDC/implementations/sweeper\_classes/boris\_2nd\_order.py                                         |      118 |        0 |    100% |           |
| pySDC/implementations/sweeper\_classes/delta\_form.py                                               |      189 |        3 |     98% |135-136, 269 |
| pySDC/implementations/sweeper\_classes/delta\_form\_MPI.py                                          |       61 |        5 |     92% |160-161, 166, 169-170 |
| pySDC/implementations/sweeper\_classes/explicit.py                                                  |       46 |        2 |     96% |   83, 125 |
| pySDC/implementations/sweeper\_classes/generic\_implicit.py                                         |       50 |        0 |    100% |           |
| pySDC/implementations/sweeper\_classes/generic\_implicit\_MPI.py                                    |      104 |        2 |     98% |   126-127 |
| pySDC/implementations/sweeper\_classes/imex\_1st\_order.py                                          |       73 |        0 |    100% |           |
| pySDC/implementations/sweeper\_classes/imex\_1st\_order\_MPI.py                                     |       40 |        0 |    100% |           |
| pySDC/implementations/sweeper\_classes/imex\_1st\_order\_mass.py                                    |       51 |        2 |     96% |   112-113 |
| pySDC/implementations/sweeper\_classes/multi\_implicit.py                                           |       59 |        8 |     86% |26, 28, 90, 151-156 |
| pySDC/implementations/sweeper\_classes/verlet.py                                                    |       73 |        9 |     88% |76, 196-205 |
| pySDC/implementations/transfer\_classes/BaseTransferDelta.py                                        |       64 |        1 |     98% |       139 |
| pySDC/implementations/transfer\_classes/BaseTransferDeltaMPI.py                                     |       40 |        4 |     90% | 63, 85-87 |
| pySDC/implementations/transfer\_classes/BaseTransferMPI.py                                          |       74 |        0 |    100% |           |
| pySDC/implementations/transfer\_classes/BaseTransfer\_mass.py                                       |       84 |       84 |      0% |     1-187 |
| pySDC/implementations/transfer\_classes/TransferFenicsMesh.py                                       |       29 |        0 |    100% |           |
| pySDC/implementations/transfer\_classes/TransferFiredrakeMesh.py                                    |       47 |        0 |    100% |           |
| pySDC/implementations/transfer\_classes/TransferMesh.py                                             |      103 |       19 |     82% |134, 186-194, 221-229 |
| pySDC/implementations/transfer\_classes/TransferMesh\_FFT2D.py                                      |       31 |        0 |    100% |           |
| pySDC/implementations/transfer\_classes/TransferMesh\_FFT.py                                        |       25 |        0 |    100% |           |
| pySDC/implementations/transfer\_classes/TransferMesh\_MPIFFT.py                                     |       78 |        0 |    100% |           |
| pySDC/implementations/transfer\_classes/TransferMesh\_NoCoarse.py                                   |        6 |        0 |    100% |           |
| pySDC/implementations/transfer\_classes/TransferPETScDMDA.py                                        |       34 |        0 |    100% |           |
| pySDC/implementations/transfer\_classes/TransferParticles\_NoCoarse.py                              |       20 |        0 |    100% |           |
| pySDC/projects/AllenCahn\_Bayreuth/AllenCahn\_dump.py                                               |       91 |       91 |      0% |     1-148 |
| pySDC/projects/AllenCahn\_Bayreuth/run\_simple\_forcing\_benchmark.py                               |       80 |        5 |     94% |30, 39, 72, 94-95 |
| pySDC/projects/AllenCahn\_Bayreuth/run\_simple\_forcing\_verification.py                            |      172 |        6 |     97% |38, 141, 287-291 |
| pySDC/projects/AllenCahn\_Bayreuth/run\_temp\_forcing\_benchmark.py                                 |       78 |       78 |      0% |     1-127 |
| pySDC/projects/AllenCahn\_Bayreuth/run\_temp\_forcing\_realistic.py                                 |       74 |       74 |      0% |     1-134 |
| pySDC/projects/AllenCahn\_Bayreuth/run\_temp\_forcing\_reference.py                                 |       74 |       74 |      0% |     1-137 |
| pySDC/projects/AllenCahn\_Bayreuth/run\_temp\_forcing\_verification.py                              |      109 |        1 |     99% |        38 |
| pySDC/projects/AllenCahn\_Bayreuth/visualize.py                                                     |       22 |       22 |      0% |      1-42 |
| pySDC/projects/AllenCahn\_Bayreuth/visualize\_temp.py                                               |       62 |       62 |      0% |     1-134 |
| pySDC/projects/AsympConv/PFASST\_conv\_Linf.py                                                      |      157 |        7 |     96% |     22-30 |
| pySDC/projects/AsympConv/PFASST\_conv\_tests.py                                                     |      156 |        0 |    100% |           |
| pySDC/projects/AsympConv/conv\_test\_to0.py                                                         |       46 |       46 |      0% |     1-101 |
| pySDC/projects/AsympConv/conv\_test\_toinf.py                                                       |       46 |       46 |      0% |      1-99 |
| pySDC/projects/AsympConv/smoother\_specrad\_heatmap.py                                              |       82 |       82 |      0% |     1-146 |
| pySDC/projects/DAE/misc/hooksDAE.py                                                                 |       19 |        0 |    100% |           |
| pySDC/projects/DAE/misc/meshDAE.py                                                                  |        3 |        0 |    100% |           |
| pySDC/projects/DAE/misc/problemDAE.py                                                               |       22 |        0 |    100% |           |
| pySDC/projects/DAE/problems/discontinuousTestDAE.py                                                 |       55 |        2 |     96% |   154-155 |
| pySDC/projects/DAE/problems/pendulum2D.py                                                           |       25 |        3 |     88% |   107-109 |
| pySDC/projects/DAE/problems/problematicF.py                                                         |       22 |        0 |    100% |           |
| pySDC/projects/DAE/problems/simpleDAE.py                                                            |       22 |        0 |    100% |           |
| pySDC/projects/DAE/problems/synchronousMachine.py                                                   |       75 |        3 |     96% |   306-308 |
| pySDC/projects/DAE/problems/transistorAmplifier.py                                                  |       59 |        2 |     97% |  134, 275 |
| pySDC/projects/DAE/problems/wscc9BusSystem.py                                                       |      214 |        1 |     99% |      1233 |
| pySDC/projects/DAE/run/accuracy\_check\_MPI.py                                                      |       56 |        0 |    100% |           |
| pySDC/projects/DAE/run/fully\_implicit\_dae\_playground.py                                          |       49 |        0 |    100% |           |
| pySDC/projects/DAE/run/run\_convergence\_test.py                                                    |       60 |        0 |    100% |           |
| pySDC/projects/DAE/run/run\_iteration\_test.py                                                      |       64 |        0 |    100% |           |
| pySDC/projects/DAE/run/synchronous\_machine\_playground.py                                          |       56 |        0 |    100% |           |
| pySDC/projects/DAE/sweepers/fullyImplicitDAE.py                                                     |       77 |        2 |     97% |   164-165 |
| pySDC/projects/DAE/sweepers/fullyImplicitDAEMPI.py                                                  |       58 |        2 |     97% |     55-56 |
| pySDC/projects/DAE/sweepers/rungeKuttaDAE.py                                                        |       54 |        0 |    100% |           |
| pySDC/projects/DAE/sweepers/semiImplicitDAE.py                                                      |       52 |        1 |     98% |        90 |
| pySDC/projects/DAE/sweepers/semiImplicitDAEMPI.py                                                   |       31 |        0 |    100% |           |
| pySDC/projects/FastWaveSlowWave/AcousticAdvection\_1D\_FD\_imex\_multiscale.py                      |       13 |        0 |    100% |           |
| pySDC/projects/FastWaveSlowWave/HookClass\_acoustic.py                                              |       12 |        0 |    100% |           |
| pySDC/projects/FastWaveSlowWave/plot\_dispersion.py                                                 |      115 |        3 |     97% | 28, 32-33 |
| pySDC/projects/FastWaveSlowWave/plot\_stab\_vs\_k.py                                                |       62 |        3 |     95% |     83-85 |
| pySDC/projects/FastWaveSlowWave/plot\_stability.py                                                  |       75 |        4 |     95% | 87, 91-93 |
| pySDC/projects/FastWaveSlowWave/plot\_stifflimit\_specrad.py                                        |       82 |        4 |     95% |     85-89 |
| pySDC/projects/FastWaveSlowWave/plotgmrescounter\_boussinesq.py                                     |       35 |        0 |    100% |           |
| pySDC/projects/FastWaveSlowWave/runconvergence\_acoustic.py                                         |      115 |       57 |     50% |    25-116 |
| pySDC/projects/FastWaveSlowWave/rungmrescounter\_boussinesq.py                                      |      107 |        0 |    100% |           |
| pySDC/projects/FastWaveSlowWave/runitererror\_acoustic.py                                           |       83 |        0 |    100% |           |
| pySDC/projects/FastWaveSlowWave/runmultiscale\_acoustic.py                                          |       97 |        1 |     99% |       143 |
| pySDC/projects/GPU/ac\_fft.py                                                                       |       48 |        0 |    100% |           |
| pySDC/projects/GPU/configs/RBC\_configs.py                                                          |      229 |      157 |     31% |8-23, 97-134, 139-150, 153-161, 166-186, 189-197, 202-213, 220-233, 236-238, 244-256, 259-267, 274-286, 293-311, 314-318, 326-353, 356-362, 365-371 |
| pySDC/projects/GPU/configs/base\_config.py                                                          |      177 |       29 |     84% |9, 11, 34-46, 75, 85, 116-120, 143, 162, 165-168, 195, 211, 243-244, 257-259, 274, 276 |
| pySDC/projects/GPU/heat.py                                                                          |        5 |        0 |    100% |           |
| pySDC/projects/GPU/paper\_plots.py                                                                  |        2 |        2 |      0% |       3-4 |
| pySDC/projects/GPU/run\_experiment.py                                                               |       48 |       23 |     52% |2-38, 58-60, 66-67 |
| pySDC/projects/Hamiltonian/fput.py                                                                  |      137 |        0 |    100% |           |
| pySDC/projects/Hamiltonian/hamiltonian\_and\_energy\_output.py                                      |       31 |        0 |    100% |           |
| pySDC/projects/Hamiltonian/hamiltonian\_output.py                                                   |       24 |        0 |    100% |           |
| pySDC/projects/Hamiltonian/harmonic\_oscillator.py                                                  |       88 |       88 |      0% |     1-161 |
| pySDC/projects/Hamiltonian/simple\_problems.py                                                      |      128 |        0 |    100% |           |
| pySDC/projects/Hamiltonian/solar\_system.py                                                         |      157 |        7 |     96% |252, 267-272 |
| pySDC/projects/Hamiltonian/stop\_at\_error\_hook.py                                                 |       12 |       12 |      0% |      1-28 |
| pySDC/projects/Monodomain/datatype\_classes/my\_mesh.py                                             |        3 |        0 |    100% |           |
| pySDC/projects/Monodomain/hooks/HookClass\_pde.py                                                   |       16 |        0 |    100% |           |
| pySDC/projects/Monodomain/hooks/HookClass\_post\_iter\_info.py                                      |       10 |        0 |    100% |           |
| pySDC/projects/Monodomain/problem\_classes/MonodomainODE.py                                         |      186 |       10 |     95% |170-181, 292, 321, 364, 376 |
| pySDC/projects/Monodomain/problem\_classes/TestODE.py                                               |       76 |        7 |     91% |40, 42, 44, 58-63 |
| pySDC/projects/Monodomain/problem\_classes/ionicmodels/cpp/\_\_init\_\_.py                          |        5 |        0 |    100% |           |
| pySDC/projects/Monodomain/problem\_classes/space\_discretizazions/Parabolic\_DCT.py                 |      149 |       26 |     83% |93-101, 130, 138, 141-146, 164, 170, 199, 222-223, 271-276, 282-283 |
| pySDC/projects/Monodomain/run\_scripts/run\_MonodomainODE.py                                        |      205 |       27 |     87% |103, 344, 355-389 |
| pySDC/projects/Monodomain/run\_scripts/run\_MonodomainODE\_cli.py                                   |       35 |        0 |    100% |           |
| pySDC/projects/Monodomain/run\_scripts/run\_TestODE.py                                              |      168 |       16 |     90% |163-166, 210-237 |
| pySDC/projects/Monodomain/sweeper\_classes/exponential\_runge\_kutta/imexexp\_1st\_order.py         |      121 |        6 |     95% |30, 277, 285, 287, 290-291 |
| pySDC/projects/Monodomain/sweeper\_classes/runge\_kutta/imexexp\_1st\_order.py                      |       54 |        2 |     96% |    26, 93 |
| pySDC/projects/Monodomain/transfer\_classes/TransferVectorOfDCTVectors.py                           |       25 |        0 |    100% |           |
| pySDC/projects/Monodomain/transfer\_classes/Transfer\_DCT\_Vector.py                                |       24 |        0 |    100% |           |
| pySDC/projects/Monodomain/utils/data\_management.py                                                 |       24 |        2 |     92% |    11, 53 |
| pySDC/projects/Performance/controller\_MPI\_scorep.py                                               |      283 |      283 |      0% |     1-637 |
| pySDC/projects/Performance/run\_simple\_forcing\_benchmark.py                                       |       83 |       83 |      0% |     1-143 |
| pySDC/projects/Performance/visualize.py                                                             |       41 |       41 |      0% |      1-97 |
| pySDC/projects/PinTSimE/battery\_model.py                                                           |      126 |        0 |    100% |           |
| pySDC/projects/PinTSimE/buck\_model.py                                                              |       13 |        0 |    100% |           |
| pySDC/projects/PinTSimE/discontinuous\_test\_ODE.py                                                 |       26 |        0 |    100% |           |
| pySDC/projects/PinTSimE/estimation\_check.py                                                        |       31 |        0 |    100% |           |
| pySDC/projects/PinTSimE/hardcoded\_solutions.py                                                     |      139 |        0 |    100% |           |
| pySDC/projects/PinTSimE/paper\_PSCC2024/log\_event.py                                               |       14 |        5 |     64% |     35-42 |
| pySDC/projects/PinTSimE/piline\_model.py                                                            |       13 |        0 |    100% |           |
| pySDC/projects/PinTSimE/switch\_estimator.py                                                        |      105 |        1 |     99% |       168 |
| pySDC/projects/RDC/equidistant\_RDC.py                                                              |       81 |        7 |     91% | 41-50, 57 |
| pySDC/projects/RDC/vanderpol\_MLSDC\_PFASST\_test.py                                                |       63 |        0 |    100% |           |
| pySDC/projects/RDC/vanderpol\_error\_test.py                                                        |       82 |       82 |      0% |     1-153 |
| pySDC/projects/RDC/vanderpol\_reference.py                                                          |       35 |       35 |      0% |      1-62 |
| pySDC/projects/RayleighBenard/RBC3D\_configs.py                                                     |      295 |       68 |     77% |7, 23, 29-30, 103-121, 136-158, 161-182, 277, 292-295, 310-313, 319-329, 335-339 |
| pySDC/projects/RayleighBenard/analysis\_scripts/RBC3D\_order.py                                     |       58 |        0 |    100% |           |
| pySDC/projects/RayleighBenard/analysis\_scripts/plot\_Nu.py                                         |       13 |        0 |    100% |           |
| pySDC/projects/RayleighBenard/analysis\_scripts/plotting\_utils.py                                  |        8 |        0 |    100% |           |
| pySDC/projects/RayleighBenard/analysis\_scripts/process\_RBC3D\_data.py                             |      134 |        7 |     95% |80, 108-109, 115, 206-208 |
| pySDC/projects/RayleighBenard/run\_experiment.py                                                    |       54 |       26 |     52% |2-38, 53, 61, 64-66, 72-73, 81 |
| pySDC/projects/RayleighBenard/sweepers.py                                                           |       39 |        2 |     95% |    36, 90 |
| pySDC/projects/Resilience/AC.py                                                                     |       98 |       25 |     74% |39, 87-88, 90-91, 97-105, 127, 160-165, 173-175, 180-183 |
| pySDC/projects/Resilience/FDeigenvalues.py                                                          |       10 |        0 |    100% |           |
| pySDC/projects/Resilience/GS.py                                                                     |       92 |       22 |     76% |25-26, 42-54, 149-154, 163-165, 170-173 |
| pySDC/projects/Resilience/Lorenz.py                                                                 |       71 |        4 |     94% |   112-115 |
| pySDC/projects/Resilience/RBC.py                                                                    |      113 |       38 |     66% |23-24, 40-52, 119, 146-151, 160-162, 167-170, 175-179, 251-262 |
| pySDC/projects/Resilience/Schroedinger.py                                                           |       82 |       11 |     87% |99-100, 128-129, 190-193, 199-202 |
| pySDC/projects/Resilience/accuracy\_check.py                                                        |      142 |       29 |     80% |120, 151, 191-199, 222, 248, 278-279, 312-325, 397-415, 438-439 |
| pySDC/projects/Resilience/advection.py                                                              |       59 |       12 |     80% |12-20, 114-121 |
| pySDC/projects/Resilience/collocation\_adaptivity.py                                                |      132 |        6 |     95% |305-309, 313-315 |
| pySDC/projects/Resilience/dahlquist.py                                                              |      147 |       21 |     86% |91, 143, 160, 222, 227-231, 286, 314-324 |
| pySDC/projects/Resilience/extrapolation\_within\_Q.py                                               |       53 |       53 |      0% |     1-121 |
| pySDC/projects/Resilience/fault\_injection.py                                                       |      184 |       16 |     91% |74, 205-206, 256, 389, 409, 414-419, 443, 467, 491-492, 541 |
| pySDC/projects/Resilience/fault\_stats.py                                                           |      451 |      199 |     56% |87-90, 137-145, 175, 245, 258, 277-279, 333-334, 370, 387-392, 516-576, 593-599, 626, 628, 630, 633-642, 654, 658, 745-749, 765-768, 783, 799-802, 1177-1181, 1187, 1191, 1193, 1195, 1239-1240, 1385, 1407, 1508-1531, 1558-1589, 1593-1662, 1666-1765 |
| pySDC/projects/Resilience/heat.py                                                                   |       35 |        1 |     97% |        71 |
| pySDC/projects/Resilience/hook.py                                                                   |       29 |        9 |     69% |60-62, 79-102 |
| pySDC/projects/Resilience/paper\_plots.py                                                           |       37 |       37 |      0% |     2-820 |
| pySDC/projects/Resilience/piline.py                                                                 |      141 |       19 |     87% |71, 93-97, 163, 289-297, 323-328 |
| pySDC/projects/Resilience/quench.py                                                                 |      217 |      111 |     49% |105-106, 154-159, 167-169, 174-177, 269, 379-442, 446-482, 486-553 |
| pySDC/projects/Resilience/reachTendExactly.py                                                       |       19 |        0 |    100% |           |
| pySDC/projects/Resilience/strategies.py                                                             |     1011 |      292 |     71% |53-54, 124, 126, 128, 131-136, 158, 161, 164-167, 180, 192, 207-210, 212, 215-224, 342-353, 389, 419-421, 425, 462, 471-488, 509, 534, 579, 603, 645, 655-656, 658, 660, 662-674, 696-703, 707, 719-728, 741-768, 783-789, 812, 829, 831, 835, 870-873, 894, 923-943, 964, 1008-1009, 1012, 1014, 1019-1025, 1028, 1030, 1066-1083, 1104, 1152-1153, 1155-1156, 1161-1162, 1197, 1218, 1234, 1255, 1267, 1288, 1313, 1371, 1384-1388, 1413, 1475, 1487-1496, 1500, 1511-1532, 1557, 1560-1563, 1623, 1636-1640, 1659-1663, 1667, 1680-1683, 1692, 1714, 1741, 1760, 1793, 1803, 1848, 1858, 1871-1886, 1901-1907, 1950-1951, 1953-1954, 1959-1960, 2001, 2058-2063, 2065, 2069, 2080-2088, 2123-2157, 2171-2176, 2197, 2201 |
| pySDC/projects/Resilience/sweepers.py                                                               |      107 |        8 |     93% |34-40, 112, 216 |
| pySDC/projects/Resilience/vdp.py                                                                    |      195 |       35 |     82% |28-54, 181-183, 189-192, 265-267, 334, 363-371, 454 |
| pySDC/projects/Resilience/work\_precision.py                                                        |      486 |      134 |     72% |75-76, 120-121, 133, 163-168, 176-177, 279, 295, 299, 301, 303-305, 309-316, 325-328, 330-331, 333-334, 336-337, 373, 377, 433-438, 734-769, 940-943, 947-954, 973, 992-1050, 1075-1076, 1138-1139, 1783-1815, 1943-1950 |
| pySDC/projects/SDC\_showdown/SDC\_timing\_Fisher.py                                                 |      109 |        0 |    100% |           |
| pySDC/projects/SDC\_showdown/SDC\_timing\_GrayScott.py                                              |      146 |       30 |     79% |   218-269 |
| pySDC/projects/Second\_orderSDC/check\_data\_folder.py                                              |        4 |        4 |      0% |       1-8 |
| pySDC/projects/Second\_orderSDC/harmonic\_oscillator\_params.py                                     |       10 |        0 |    100% |           |
| pySDC/projects/Second\_orderSDC/harmonic\_oscillator\_run\_points.py                                |        3 |        3 |      0% |       1-3 |
| pySDC/projects/Second\_orderSDC/harmonic\_oscillator\_run\_stab\_interval.py                        |        3 |        3 |      0% |       1-3 |
| pySDC/projects/Second\_orderSDC/harmonic\_oscillator\_run\_stability.py                             |        2 |        2 |      0% |       1-2 |
| pySDC/projects/Second\_orderSDC/penningtrap\_HookClass.py                                           |       19 |        0 |    100% |           |
| pySDC/projects/Second\_orderSDC/penningtrap\_Simulation.py                                          |      133 |       32 |     76% |24-25, 31-34, 44-52, 115-117, 187-190, 196-197, 203-206, 212-213, 220-221 |
| pySDC/projects/Second\_orderSDC/penningtrap\_params.py                                              |       27 |        0 |    100% |           |
| pySDC/projects/Second\_orderSDC/penningtrap\_run\_Hamiltonian\_error.py                             |       10 |       10 |      0% |      2-14 |
| pySDC/projects/Second\_orderSDC/penningtrap\_run\_error.py                                          |        2 |        2 |      0% |       1-2 |
| pySDC/projects/Second\_orderSDC/penningtrap\_run\_work\_precision.py                                |        3 |        3 |      0% |       2-5 |
| pySDC/projects/Second\_orderSDC/plot\_helper.py                                                     |        4 |        0 |    100% |           |
| pySDC/projects/Second\_orderSDC/stability\_simulation.py                                            |      107 |       11 |     90% |101, 108, 254-265, 279 |
| pySDC/projects/StroemungsRaum/hooks/hooks\_NSE\_IMEX\_FEniCS.py                                     |       40 |        0 |    100% |           |
| pySDC/projects/StroemungsRaum/problem\_classes/ConvectionDiffusion\_2D\_FEniCS.py                   |       58 |        0 |    100% |           |
| pySDC/projects/StroemungsRaum/problem\_classes/HeatEquation\_2D\_FEniCS.py                          |       57 |        0 |    100% |           |
| pySDC/projects/StroemungsRaum/problem\_classes/NavierStokes\_2D\_FEniCS.py                          |       93 |        0 |    100% |           |
| pySDC/projects/StroemungsRaum/problem\_classes/NavierStokes\_2D\_TaylorGreen\_monolithic\_FEniCS.py |      136 |        0 |    100% |           |
| pySDC/projects/StroemungsRaum/problem\_classes/NavierStokes\_2D\_monolithic\_FEniCS.py              |      107 |        0 |    100% |           |
| pySDC/projects/StroemungsRaum/problem\_classes/newton\_step.py                                      |       17 |        0 |    100% |           |
| pySDC/projects/StroemungsRaum/run\_Navier\_Stokes\_TaylorGreen\_FEniCS.py                           |       69 |       13 |     81% |   235-254 |
| pySDC/projects/StroemungsRaum/run\_Navier\_Stokes\_equations\_FEniCS.py                             |       60 |       17 |     72% |   119-146 |
| pySDC/projects/StroemungsRaum/run\_Navier\_Stokes\_equations\_monolithic\_FEniCS.py                 |       36 |        0 |    100% |           |
| pySDC/projects/StroemungsRaum/run\_accuracy/run\_accuracy\_heat\_equation\_FEniCS.py                |       25 |        5 |     80% |   127-133 |
| pySDC/projects/StroemungsRaum/run\_convection\_diffusion\_equation\_FEniCS.py                       |       64 |       18 |     72% |   126-158 |
| pySDC/projects/StroemungsRaum/run\_heat\_equation\_FEniCS.py                                        |       63 |       17 |     73% |   129-159 |
| pySDC/projects/StroemungsRaum/sweepers/generic\_implicit\_mass.py                                   |       58 |        7 |     88% |28, 40, 51, 111-112, 129, 132 |
| pySDC/projects/StroemungsRaum/sweepers/imex\_1st\_order\_mass\_NSE.py                               |       25 |        2 |     92% |    39, 49 |
| pySDC/projects/TOMS/AllenCahn\_contracting\_circle.py                                               |      185 |       10 |     95% |   316-329 |
| pySDC/projects/TOMS/pySDC\_with\_PETSc.py                                                           |       83 |       83 |      0% |     1-152 |
| pySDC/projects/TOMS/visualize\_pySDC\_with\_PETSc.py                                                |       93 |        1 |     99% |        29 |
| pySDC/projects/compression/compression\_convergence\_controller.py                                  |       23 |        0 |    100% |           |
| pySDC/projects/compression/order.py                                                                 |       79 |       16 |     80% |61-62, 68-70, 131-144 |
| pySDC/projects/matrixPFASST/compare\_to\_matrixbased.py                                             |      142 |        0 |    100% |           |
| pySDC/projects/matrixPFASST/compare\_to\_propagator.py                                              |      135 |        0 |    100% |           |
| pySDC/projects/matrixPFASST/controller\_matrix\_nonMPI.py                                           |      183 |        4 |     98% |204-205, 218, 258 |
| pySDC/projects/parallelSDC/AllenCahn\_parallel.py                                                   |       92 |        0 |    100% |           |
| pySDC/projects/parallelSDC/ErrReductionHook.py                                                      |       25 |        0 |    100% |           |
| pySDC/projects/parallelSDC/GeneralizedFisher\_1D\_FD\_implicit\_Jac.py                              |       12 |        0 |    100% |           |
| pySDC/projects/parallelSDC/Van\_der\_Pol\_implicit\_Jac.py                                          |       14 |       14 |      0% |      1-45 |
| pySDC/projects/parallelSDC/linearized\_implicit\_fixed\_parallel.py                                 |       37 |        1 |     97% |        26 |
| pySDC/projects/parallelSDC/linearized\_implicit\_fixed\_parallel\_prec.py                           |       10 |        1 |     90% |        24 |
| pySDC/projects/parallelSDC/linearized\_implicit\_parallel.py                                        |       37 |        1 |     97% |        21 |
| pySDC/projects/parallelSDC/minimization.py                                                          |       48 |       48 |      0% |      1-71 |
| pySDC/projects/parallelSDC/newton\_vs\_sdc.py                                                       |       97 |        0 |    100% |           |
| pySDC/projects/parallelSDC/nonlinear\_playground.py                                                 |      106 |        0 |    100% |           |
| pySDC/projects/parallelSDC/preconditioner\_playground.py                                            |      142 |        4 |     97% |126-127, 227-228 |
| pySDC/projects/parallelSDC/preconditioner\_playground\_MPI.py                                       |      151 |        4 |     97% |133-134, 238-239 |
| pySDC/projects/parallelSDC\_reloaded/allenCahn\_accuracy.py                                         |       59 |        0 |    100% |           |
| pySDC/projects/parallelSDC\_reloaded/allenCahn\_setup.py                                            |       26 |        0 |    100% |           |
| pySDC/projects/parallelSDC\_reloaded/chemicalReaction\_accuracy.py                                  |       55 |        0 |    100% |           |
| pySDC/projects/parallelSDC\_reloaded/chemicalReaction\_setup.py                                     |       24 |        0 |    100% |           |
| pySDC/projects/parallelSDC\_reloaded/convergence.py                                                 |       38 |        0 |    100% |           |
| pySDC/projects/parallelSDC\_reloaded/jacobiElliptic\_accuracy.py                                    |       53 |        0 |    100% |           |
| pySDC/projects/parallelSDC\_reloaded/jacobiElliptic\_setup.py                                       |       29 |        0 |    100% |           |
| pySDC/projects/parallelSDC\_reloaded/kaps\_accuracy.py                                              |       57 |        0 |    100% |           |
| pySDC/projects/parallelSDC\_reloaded/kaps\_setup.py                                                 |       23 |        0 |    100% |           |
| pySDC/projects/parallelSDC\_reloaded/lorenz\_accuracy.py                                            |       58 |        0 |    100% |           |
| pySDC/projects/parallelSDC\_reloaded/lorenz\_setup.py                                               |       24 |        0 |    100% |           |
| pySDC/projects/parallelSDC\_reloaded/nilpotency.py                                                  |       46 |        0 |    100% |           |
| pySDC/projects/parallelSDC\_reloaded/protheroRobinsonAutonomous\_accuracy.py                        |       59 |        0 |    100% |           |
| pySDC/projects/parallelSDC\_reloaded/protheroRobinsonAutonomous\_setup.py                           |       30 |        0 |    100% |           |
| pySDC/projects/parallelSDC\_reloaded/protheroRobinson\_accuracy.py                                  |       59 |        0 |    100% |           |
| pySDC/projects/parallelSDC\_reloaded/protheroRobinson\_setup.py                                     |       30 |        0 |    100% |           |
| pySDC/projects/parallelSDC\_reloaded/scripts/fig01\_conv.py                                         |       41 |        0 |    100% |           |
| pySDC/projects/parallelSDC\_reloaded/scripts/fig02\_stab.py                                         |       40 |        0 |    100% |           |
| pySDC/projects/parallelSDC\_reloaded/scripts/fig03\_lorenz.py                                       |      102 |        0 |    100% |           |
| pySDC/projects/parallelSDC\_reloaded/scripts/fig04\_protheroRobinson.py                             |       62 |        0 |    100% |           |
| pySDC/projects/parallelSDC\_reloaded/scripts/fig05\_allenCahn.py                                    |       76 |        0 |    100% |           |
| pySDC/projects/parallelSDC\_reloaded/scripts/fig06\_allenCahnMPI.py                                 |       64 |        3 |     95% | 48, 61-62 |
| pySDC/projects/parallelSDC\_reloaded/scripts/fig06\_allenCahnMPI\_plot.py                           |       55 |        0 |    100% |           |
| pySDC/projects/parallelSDC\_reloaded/stability.py                                                   |       35 |        0 |    100% |           |
| pySDC/projects/parallelSDC\_reloaded/utils.py                                                       |      168 |        2 |     99% |   311-312 |
| pySDC/projects/parallelSDC\_reloaded/vanderpol\_accuracy.py                                         |       62 |        0 |    100% |           |
| pySDC/projects/parallelSDC\_reloaded/vanderpol\_setup.py                                            |       36 |        0 |    100% |           |
| pySDC/projects/soft\_failure/FaultHooks.py                                                          |       20 |        0 |    100% |           |
| pySDC/projects/soft\_failure/generate\_statistics.py                                                |      206 |       49 |     76% |26-63, 71-111, 169, 171, 211, 213 |
| pySDC/projects/soft\_failure/implicit\_sweeper\_faults.py                                           |      144 |        5 |     97% |44, 159-161, 258 |
| pySDC/projects/soft\_failure/visualization\_helper.py                                               |       54 |        0 |    100% |           |
| pySDC/tutorial/step\_1/A\_spatial\_problem\_setup.py                                                |       21 |        0 |    100% |           |
| pySDC/tutorial/step\_1/B\_spatial\_accuracy\_check.py                                               |       81 |        0 |    100% |           |
| pySDC/tutorial/step\_1/C\_collocation\_problem\_setup.py                                            |       26 |        0 |    100% |           |
| pySDC/tutorial/step\_1/D\_collocation\_accuracy\_check.py                                           |       86 |        0 |    100% |           |
| pySDC/tutorial/step\_2/A\_step\_data\_structure.py                                                  |       38 |        0 |    100% |           |
| pySDC/tutorial/step\_2/B\_my\_first\_sweeper.py                                                     |       54 |        0 |    100% |           |
| pySDC/tutorial/step\_2/C\_using\_pySDCs\_frontend.py                                                |       43 |        0 |    100% |           |
| pySDC/tutorial/step\_3/A\_getting\_statistics.py                                                    |       54 |        0 |    100% |           |
| pySDC/tutorial/step\_3/B\_adding\_statistics.py                                                     |       55 |        0 |    100% |           |
| pySDC/tutorial/step\_3/C\_study\_collocations.py                                                    |       63 |        0 |    100% |           |
| pySDC/tutorial/step\_3/HookClass\_Particles.py                                                      |       49 |        0 |    100% |           |
| pySDC/tutorial/step\_4/A\_spatial\_transfer\_operators.py                                           |       45 |        0 |    100% |           |
| pySDC/tutorial/step\_4/B\_multilevel\_hierarchy.py                                                  |       43 |        0 |    100% |           |
| pySDC/tutorial/step\_4/C\_SDC\_vs\_MLSDC.py                                                         |       80 |        0 |    100% |           |
| pySDC/tutorial/step\_4/D\_MLSDC\_with\_particles.py                                                 |       78 |        0 |    100% |           |
| pySDC/tutorial/step\_4/PenningTrap\_3D\_coarse.py                                                   |       11 |        0 |    100% |           |
| pySDC/tutorial/step\_5/A\_multistep\_multilevel\_hierarchy.py                                       |       32 |        0 |    100% |           |
| pySDC/tutorial/step\_5/B\_my\_first\_PFASST\_run.py                                                 |       72 |        0 |    100% |           |
| pySDC/tutorial/step\_5/C\_advection\_and\_PFASST.py                                                 |       81 |        0 |    100% |           |
| pySDC/tutorial/step\_6/A\_run\_non\_MPI\_controller.py                                              |       97 |        0 |    100% |           |
| pySDC/tutorial/step\_6/B\_odd\_temporal\_distribution.py                                            |        3 |        0 |    100% |           |
| pySDC/tutorial/step\_6/C\_MPI\_parallelization.py                                                   |       37 |        0 |    100% |           |
| pySDC/tutorial/step\_7/A\_pySDC\_with\_FEniCS.py                                                    |      105 |        0 |    100% |           |
| pySDC/tutorial/step\_7/B\_pySDC\_with\_mpi4pyfft.py                                                 |       88 |        0 |    100% |           |
| pySDC/tutorial/step\_7/C\_pySDC\_with\_PETSc.py                                                     |       92 |        2 |     98% |    35, 43 |
| pySDC/tutorial/step\_7/D\_pySDC\_with\_PyTorch.py                                                   |       42 |        0 |    100% |           |
| pySDC/tutorial/step\_7/E\_pySDC\_with\_Firedrake.py                                                 |      105 |        0 |    100% |           |
| pySDC/tutorial/step\_7/F\_pySDC\_with\_Gusto.py                                                     |      112 |       10 |     91% |137-142, 206-213, 341 |
| pySDC/tutorial/step\_7/G\_pySDC\_on\_GPU.py                                                         |       43 |        0 |    100% |           |
| pySDC/tutorial/step\_7/torch\_heat.py                                                               |      133 |       28 |     79% |15-16, 77-81, 99, 113, 126-127, 211-212, 241, 259-262, 280-282, 289-297, 360 |
| pySDC/tutorial/step\_8/A\_visualize\_residuals.py                                                   |       32 |        0 |    100% |           |
| pySDC/tutorial/step\_8/B\_multistep\_SDC.py                                                         |       98 |        0 |    100% |           |
| pySDC/tutorial/step\_8/C\_iteration\_estimator.py                                                   |      179 |        0 |    100% |           |
| pySDC/tutorial/step\_8/HookClass\_error\_output.py                                                  |       30 |        0 |    100% |           |
| pySDC/tutorial/step\_9/A\_paradiag\_for\_linear\_problems.py                                        |      111 |        0 |    100% |           |
| pySDC/tutorial/step\_9/B\_paradiag\_for\_nonlinear\_problems.py                                     |       83 |        1 |     99% |       109 |
| pySDC/tutorial/step\_9/C\_paradiag\_in\_pySDC.py                                                    |       88 |        0 |    100% |           |
| pySDC/tutorial/step\_9/D\_adaptive\_alpha.py                                                        |       77 |        0 |    100% |           |
| pySDC/tutorial/step\_9/E\_paradiag\_MPI.py                                                          |       16 |        0 |    100% |           |
| **TOTAL**                                                                                           | **32540** | **4221** | **87%** |           |

42 empty files skipped.


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/Parallel-in-Time/pySDC/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/Parallel-in-Time/pySDC/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Parallel-in-Time/pySDC/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/Parallel-in-Time/pySDC/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2FParallel-in-Time%2FpySDC%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/Parallel-in-Time/pySDC/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.