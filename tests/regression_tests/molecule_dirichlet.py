import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy
import pickle

from pygbe.util import an_solution
from regression import (scanOutput, run_regression, picklesave, pickleload,
                        report_results, mesh)


def main():

    print('{:-^60}'.format('Running molecule_dirichlet test'))

    try:
        test_outputs = pickleload()
    except IOError:
        test_outputs = {}

    problem_folder = 'input_files'

    #molecule_dirichlet
    param = 'sphere_fine.param'
    test_name = 'molecule_dirichlet'
    if test_name not in test_outputs.keys():
        N, iterations, Esolv, Esurf, Ecoul, Time = run_regression(
            mesh, test_name, problem_folder, param)
        test_outputs[test_name] = [N, iterations, Esolv, Esurf, Ecoul, Time]

    picklesave(test_outputs)

    #molecule_single_center
    param = 'sphere_fine.param'
    test_name = 'molecule_single_center'
    if test_name not in test_outputs.keys():
        N, iterations, Esolv, Esurf, Ecoul, Time = run_regression(
            mesh, test_name, problem_folder, param)
        test_outputs[test_name] = [N, iterations, Esolv, Esurf, Ecoul, Time]

    picklesave(test_outputs)

    #dirichlet_surface
    param = 'sphere_fine.param'
    test_name = 'dirichlet_surface'
    if test_name not in test_outputs.keys():
        N, iterations, Esolv, Esurf, Ecoul, Time = run_regression(
            mesh, test_name, problem_folder, param)
        test_outputs[test_name] = [N, iterations, Esolv, Esurf, Ecoul, Time]

    picklesave(test_outputs)

    Esolv, Esurf, Ecoul = test_outputs['molecule_dirichlet'][2:5]
    Esolv_mol, Esurf_mol, Ecoul_mol = test_outputs['molecule_single_center'][2:
                                                                             5]
    Esolv_surf, Esurf_surf, Ecoul_surf = test_outputs['dirichlet_surface'][2:5]
    Time = test_outputs['molecule_dirichlet'][-1]
    Time_mol = test_outputs['molecule_single_center'][-1]
    Time_surf = test_outputs['dirichlet_surface'][-1]
    N, iterations = test_outputs['molecule_dirichlet'][:2]

    Einter = Esolv + Esurf + Ecoul - Esolv_surf - Esurf_mol - Ecoul_mol - Esolv_mol - Esurf_surf - Ecoul_surf
    total_time = Time + Time_mol + Time_surf

    analytical = an_solution.molecule_constant_potential(1., 1., 5., 4., 12.,
                                                         0.125, 4., 80.)

    error = abs(Einter - analytical) / abs(analytical)

    report_results(error, N, iterations, Einter, analytical, total_time)


if __name__ == "__main__":
    from check_for_meshes import check_mesh
    check_mesh()
    main()
