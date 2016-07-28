"""
It contains the necessary functions to set up the surface to be solved.
"""

import time
import numpy
from scipy import linalg

from pygbe.tree.FMMutils import addSources, sortPoints, generateTree, findTwigs
from pygbe.tree.direct import computeDiagonal
from pygbe.util.semi_analytical import GQ_1D
from pygbe.util.readData import (readVertex, readTriangle, readpqr, readcrd,
                           readFields, read_surface)
from pygbe.quadrature import quadratureRule_fine, getGaussPoints
from pygbe.classes import Surface, Field


def initializeSurf(field_array, filename, param):
    """
    Initialize the surface of the molecule.

    Arguments
    ----------
    field_array: array, contains the Field classes of each region on the surface.
    filename   : name of the file that contains the surface information.
    param      : class, parameters related to the surface.

    Returns
    --------
    surf_array : array, contains the surface classes of each region on the
                        surface.
    """

    surf_array = []

    # Read filenames for surfaces
    files, surf_type, phi0_file = read_surface(filename)
    Nsurf = len(files)

    for i in range(Nsurf):
        print('\nReading surface {} from file {}'.format(i, files[i]))

        s = {}#Surface()

        s['surf_type'] = surf_type[i]

        if s['surf_type'] == 'dirichlet_surface' or s['surf_type'] == 'neumann_surface':
            s['phi0'] = numpy.loadtxt(phi0_file[i])
            print('\nReading phi0 file for surface {} from {}'.format(i, phi0_file[i]))

        Area_null = []
        tic = time.time()
        s['vertex'] = readVertex(files[i] + '.vert', param.REAL)
        triangle_raw = readTriangle(files[i] + '.face', s['surf_type'])
        toc = time.time()
        print('Time load mesh: {}'.format(toc - tic))
        Area_null = zero_areas(s, triangle_raw, Area_null)
        s['triangle'] = numpy.delete(triangle_raw, Area_null, 0)
        print('Removed areas=0: {}'.format(len(Area_null)))

        # Look for regions inside/outside
        for j in range(Nsurf + 1):
            if len(field_array[j].parent) > 0:
                if field_array[j].parent[0] == i:  # Inside region
                    s['kappa_in'] = field_array[j].kappa
                    s['Ein'] = field_array[j].E
                    s['LorY_in'] = field_array[j].LorY
            if len(field_array[j].child) > 0:
                if i in field_array[j].child:  # Outside region
                    s['kappa_out'] = field_array[j].kappa
                    s['Eout'] = field_array[j].E
                    s['LorY_out'] = field_array[j].LorY

        if s['surf_type'] != 'dirichlet_surface' and s['surf_type'] != 'neumann_surface':
            s['E_hat'] = s['Ein'] / s['Eout']
        else:
            s['E_hat'] = 1

        surf_array.append(s)
    return surf_array


def zero_areas(s, triangle_raw, Area_null):
    """
    Looks for "zero-areas", areas that are really small, almost zero. It appends
    them to Area_null list.

    Arguments
    ----------
    s           : class, surface where we whan to look for zero areas.
    triangle_raw: list, triangles of the surface.
    Area_null   : list, contains the zero areas.

    Returns
    --------
    Area_null   : list, indices of the triangles with zero-areas.
    """

    for i in range(len(triangle_raw)):
        L0 = s['vertex'][triangle_raw[i, 1]] - s['vertex'][triangle_raw[i, 0]]
        L2 = s['vertex'][triangle_raw[i, 0]] - s['vertex'][triangle_raw[i, 2]]
        normal_aux = numpy.cross(L0, L2)
        Area_aux = linalg.norm(normal_aux) / 2
        if Area_aux < 1e-10:
            Area_null.append(i)

    return Area_null


def fill_surface(surf, param):
    """
    It fills the surface with all the necessary information to solve it.

    -It sets the Gauss points.
    -It generates tree, computes the indices and precompute terms for M2M.
    -It generates preconditioner.
    -It computes the diagonal integral for internal and external equations.

    Arguments
    ----------
    surf     : class, surface that we are studying.
    param    : class, parameters related to the surface we are studying.

    Returns
    --------
    time_sort: float, time spent in sorting the data needed for the treecode.
    """

    N = len(surf['triangle'])
    Nj = N * param.K
    # Calculate centers
    surf['center_coords'] = numpy.average(surf['vertex'][surf['triangle'][:], :], axis=1)
#    surf.center_coords = dict(zip(keys, [_coords[:,0], _coords[:, 1], _coords[:,2]]))
#    surf.center_coords['x'] = numpy.average(surf.vertex[surf.triangle[:], 0], axis=1)
#    surf.center_coords['y'] = numpy.average(surf.vertex[surf.triangle[:], 1], axis=1)
#    surf.center_coords['z'] = numpy.average(surf.vertex[surf.triangle[:], 2], axis=1)

#    surf['normal'] = numpy.zeros((N, 3))
#    surf['area'] = numpy.zeros((N, 1))

    L0 = surf['vertex'][surf['triangle'][:, 1]] - surf['vertex'][surf['triangle'][:, 0]]
    L2 = surf['vertex'][surf['triangle'][:, 0]] - surf['vertex'][surf['triangle'][:, 2]]
    surf['normal'] = numpy.cross(L0, L2)
    surf['area'] = numpy.sqrt((surf['normal']**2).sum(axis=1)) / 2
#    surf['area'] = surf['area'].reshape(surf['area'].shape[0], -1)
#    numpy.sqrt(surf.normal[:, 0]**2 + surf.normal[:, 1]**2 +
#                           surf['normal'][:, 2]**2) / 2

    #newaxis adds a dimension to an existing array (so (n,) -> (n, 1))
    surf['normal'] = surf['normal'] / (2 * surf['area'][:, numpy.newaxis])
#    surf.normal[:, 0] = surf.normal[:, 0] / (2 * surf.Area)
#    surf.normal[:, 1] = surf.normal[:, 1] / (2 * surf.Area)
#    surf.normal[:, 2] = surf.normal[:, 2] / (2 * surf.Area)

    # Set Gauss points (sources)
    surf['gauss_coords'] = getGaussPoints(surf['vertex'],
                                          surf['triangle'],
                                          param.K)

   # dict(zip(keys,
   #                              getGaussPoints(surf.vertex,
   #                                             surf.triangle,
   #                                             param.K)))
#    surf.xj, surf.yj, surf.zj = getGaussPoints(surf.vertex, surf.triangle,
#                                               param.K)

    xcenter = numpy.average(surf['center_coords'], axis=0)
#    x_center = numpy.zeros(3)
#    x_center[0] = numpy.average(surf.xi).astype(param.REAL)
#    x_center[1] = numpy.average(surf.yi).astype(param.REAL)
#    x_center[2] = numpy.average(surf.zi).astype(param.REAL)
#    dist = numpy.sqrt((surf.xi - x_center[0])**2 + (surf.yi - x_center[1])**2 +
#                      (surf.zi - x_center[2])**2)
    R_C0 = numpy.sqrt(((surf['center_coords'] - xcenter)**2).sum(axis=1)).max()

    # Generate tree, compute indices and precompute terms for M2M
    surf['tree'] = generateTree(surf['center_coords'], param.NCRIT, param.Nm,
                             N, R_C0, x_center)
    C = 0
    surf['twig'] = findTwigs(surf['tree'], C, surf['twig'], param.NCRIT)

    addSources(surf['tree'], surf['twig'], param.K)

    surf['xk'], surf['wk'] = GQ_1D(param.Nk)
    surf['Xsk'], surf['Wsk'] = quadratureRule_fine(param.K_fine)

#    # Stores the inverse of the block diagonal (also a tridiag matrix)
#    # Order: Top left, top right, bott left, bott right
#    centers = numpy.zeros((N, 3))
#    centers[:, 0] = surf.xi[:]
#    centers[:, 1] = surf.yi[:]
#    centers[:, 2] = surf.zi[:]

    #   Compute diagonal integral for internal equation
    VL = numpy.zeros(N)
    KL = numpy.zeros(N)
    VY = numpy.zeros(N)
    KY = numpy.zeros(N)
    computeDiagonal(VL, KL, VY, KY, numpy.ravel(surf['vertex'][surf['triangle'][:]]),
                    numpy.ravel(centers), surf['kappa_in'], 2 * numpy.pi, 0.,
                    surf['xk'], surf['wk'])
    if surf['LorY_in'] == 1:
        dX11 = KL
        dX12 = -VL
        surf['sglInt_int'] = VL  # Array for singular integral of V through interior
    elif surf['LorY_in'] == 2:
        dX11 = KY
        dX12 = -VY
        surf['sglInt_int'] = VY  # Array for singular integral of V through interior
    else:
        surf['sglInt_int'] = numpy.zeros(N)

#   Compute diagonal integral for external equation
    VL = numpy.zeros(N)
    KL = numpy.zeros(N)
    VY = numpy.zeros(N)
    KY = numpy.zeros(N)
    computeDiagonal(VL, KL, VY, KY, numpy.ravel(surf['vertex'][surf['triangle'][:]]),
                    numpy.ravel(centers), surf['kappa_out'], 2 * numpy.pi, 0.,
                    surf['xk'], surf['wk'])
    if surf['LorY_out'] == 1:
        dX21 = KL
        dX22 = surf['E_hat'] * VL
        surf['sglInt_ext'] = VL  # Array for singular integral of V through exterior
    elif surf['LorY_out'] == 2:
        dX21 = KY
        dX22 = surf['E_hat'] * VY
        surf['sglInt_ext'] = VY  # Array for singular integral of V through exterior
    else:
        surf['sglInt_ext'] = numpy.zeros(N)

    # Generate preconditioner
    # Will use block-diagonal preconditioner (AltmanBardhanWhiteTidor2008)
    #If we have complex dielectric constants we need to initialize Precon with
    #complex type else it'll be float.
    if type(surf['E_hat']) == complex:
        surf['Precond'] = numpy.zeros((4, N), complex)
    else:
        surf['Precond'] = numpy.zeros((4, N))

    if (surf['surf_type'] != 'dirichlet_surface'
        and surf['surf_type'] != 'neumann_surface'):
        d_aux = 1 / (dX22 - dX21 * dX12 / dX11)
        surf['Precond'][0, :] = 1 / dX11 + 1 / dX11 * dX12 * d_aux * dX21 / dX11
        surf['Precond'][1, :] = -1 / dX11 * dX12 * d_aux
        surf['Precond'][2, :] = -d_aux * dX21 / dX11
        surf['Precond'][3, :] = d_aux
    elif surf['surf_type'] == 'dirichlet_surface':
        surf['Precond'][0, :] = 1 / VY  # So far only for Yukawa outside
    elif surf['surf_type'] == 'neumann_surface' or surf['surf_type'] == 'asc_surface':
        surf['Precond'][0, :] = 1 / (2 * numpy.pi)

    tic = time.time()
    sortPoints(surf, surf['tree'], surf['twig'], param)
    toc = time.time()
    time_sort = toc - tic

    return time_sort


#def compute_diagonal_integral(surf, internal=True):
#    """
#
#    Computes a diagonal integral for either the internal or external equation
#
#    Arguments
#    ---------
#    surf  :  dictionary, surface parameters
#    internal : kwarg to determine if internal or external surface
#               should be evaluated
#
#    Returns
#    _______
#    surf  :  dictionary, modified in place
#
#    """
#
#    VL = numpy.zeros(N)
#    KL = numpy.zeros(N)
#    VY = numpy.zeros(N)
#    KY = numpy.zeros(N)
#
#    if internal:
#
#    kappa = 'kappa_{}'.format(side)
#    LorY = 'LorY_{}'.format(side)
#    sglInt = 'sglInt_{}'.format
#    computeDiagonal(VL, KL, VY, KY, numpy.ravel(surf['vertex'][surf['triangle'][:]]),
#                    numpy.ravel(centers), surf['kappa_in'], 2 * numpy.pi, 0.,
#                    surf['xk'], surf['wk'])
#    if surf['LorY_in'] == 1:
#        dX11 = KL
#        dX12 = -VL
#        surf['sglInt_int'] = VL  # Array for singular integral of V through interior
#    elif surf['LorY_in'] == 2:
#        dX11 = KY
#        dX12 = -VY
#        surf['sglInt_int'] = VY  # Array for singular integral of V through interior
#    else:
#        surf['sglInt_int'] = numpy.zeros(N)
#
#    return surf


def initializeField(filename, param):
    """
    Initialize all the regions in the surface to be solved.

    Arguments
    ----------
    filename   : name of the file that contains the surface information.
    param      : class, parameters related to the surface.

    Returns
    --------
    field_array: array, contains the Field classes of each region on the surface.
    """

    LorY, pot, E, kappa, charges, coulomb, qfile, Nparent, parent, Nchild, child = readFields(
        filename)

    Nfield = len(LorY)
    field_array = []
    Nchild_aux = 0
    for i in range(Nfield):
        if int(pot[i]) == 1:
            param.E_field.append(i)  # This field is where the energy will be calculated
        field_aux = Field()

        try:
            field_aux.LorY = int(LorY[i])  # Laplace of Yukawa
        except ValueError:
            field_aux.LorY = 0

        if 'j' in E[i]:
            field_aux.E = complex(E[i])
        else:
            try:
                field_aux.E = param.REAL(E[i])  # Dielectric constant
            except ValueError:
                field_aux.E = 0
        try:
            field_aux.kappa = param.REAL(kappa[i])  # inverse Debye length
        except ValueError:
            field_aux.kappa = 0

        field_aux.coulomb = int(coulomb[i])  # do/don't coulomb interaction
        if int(charges[i]) == 1:  # if there are charges
            if qfile[i][-4:] == '.crd':
                xq, q, Nq = readcrd(qfile[i], param.REAL)  # read charges
                print('\nReading crd for region {} from {}'.format(i, qfile[i]))
            if qfile[i][-4:] == '.pqr':
                xq, q, Nq = readpqr(qfile[i], param.REAL)  # read charges
                print('\nReading pqr for region {} from {}'.format(i, qfile[i]))
            field_aux.xq = xq  # charges positions
            field_aux.q = q  # charges values
        if int(Nparent[i]) == 1:  # if it is an enclosed region
            field_aux.parent.append(
                int(parent[i])
            )  # pointer to parent surface (enclosing surface)
        if int(Nchild[i]) > 0:  # if there are enclosed regions inside
            for j in range(int(Nchild[i])):
                field_aux.child.append(int(child[Nchild_aux + j])
                                       )  # Loop over children to get pointers
            Nchild_aux += int(Nchild[i]) - 1  # Point to child for next surface
            Nchild_aux += 1

        field_array.append(field_aux)
    return field_array


def fill_phi(phi, surf_array):
    """
    It places the result vector on surface structure.

    Arguments
    ----------
    phi        : array, result vector.
    surf_array : array, contains the surface classes of each region on the
                        surface.
    """

    s_start = 0
    for i in range(len(surf_array)):
        s_size = len(surf_array[i].xi)
        if surf_array[i].surf_type == 'dirichlet_surface':
            surf_array[i].phi = surf_array[i].phi0
            surf_array[i].dphi = phi[s_start:s_start + s_size]
            s_start += s_size
        elif surf_array[i].surf_type == 'neumann_surface':
            surf_array[i].dphi = surf_array[i].phi0
            surf_array[i].phi = phi[s_start:s_start + s_size]
            s_start += s_size
        elif surf_array[i].surf_type == 'asc_surface':
            surf_array[i].dphi = phi[s_start:s_start + s_size] / surf_array[
                i].Ein
            surf_array[i].phi = numpy.zeros(s_size)
            s_start += s_size
        else:
            surf_array[i].phi = phi[s_start:s_start + s_size]
            surf_array[i].dphi = phi[s_start + s_size:s_start + 2 * s_size]
            s_start += 2 * s_size
