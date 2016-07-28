"""
It contains the function in charge of the data transfer to the GPU.
"""
import numpy

# PyCUDA libraries
try:
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    import pycuda.driver as cuda
except:
    pass


def data_transfer(surf_array, d_surf_array, field_array, ind, param, kernel):
    """
    It manages the data transfer to the GPU.

    Arguments
    ----------
    surf_array : array, contains the surface classes of each region on the
                        surface.
    d_surf_array : array of dicts, holding info corresponding to surf_array but on a GPU
    field_array: array, contains the Field classes of each region on the surface.
    ind        : class, it contains the indices related to the treecode
                        computation.
    param      : class, parameters related to the surface.
    kernel: pycuda source module.
    """

    REAL = param.REAL
    Nsurf = len(surf_array)
    for s in range(Nsurf):
        d_surf_array[s] = {}
        d_surf_array[s]['d_center_coords'] = gpuarray.to_gpu(surf_array[s]['center_coords_sort'])
        d_surf_array[s]['d_gauss_coords'] = gpuarray.to_gpu(surf_array[s]['gauss_coords_sort'])
#        surf_array[s].xiDev = gpuarray.to_gpu(surf_array[s].xiSort.astype(
#            REAL))
#        surf_array[s].yiDev = gpuarray.to_gpu(surf_array[s].yiSort.astype(
#            REAL))
#        surf_array[s].ziDev = gpuarray.to_gpu(surf_array[s].ziSort.astype(
#            REAL))
#        surf_array[s].xjDev = gpuarray.to_gpu(surf_array[s].xjSort.astype(
#            REAL))
#        surf_array[s].yjDev = gpuarray.to_gpu(surf_array[s].yjSort.astype(
#            REAL))
#        surf_array[s].zjDev = gpuarray.to_gpu(surf_array[s].zjSort.astype(
#            REAL))
        d_surf_array[s]['d_area_sort'] = gpuarray.to_gpu(surf_array[s]['area_sort'].astype(
              REAL))
        d_surf_array[s]['sglInt_intDev'] = gpuarray.to_gpu(surf_array[
              s]['sglInt_intSort'].astype(REAL))
        d_surf_array[s]['sglInt_extDev'] = gpuarray.to_gpu(surf_array[
              s]['sglInt_extSort'].astype(REAL))
        d_surf_array[s]['vertexDev'] = gpuarray.to_gpu(numpy.ravel(surf_array[
              s]['vertex'][surf_array[s]['triangle_sort']]).astype(REAL))

        d_surf_array[s]['d_xc'] = gpuarray.to_gpu(numpy.ravel(surf_array[s]['xc_sort']))
        d_surf_array[s]['d_yc'] = gpuarray.to_gpu(numpy.ravel(surf_array[s]['yc_sort']))
        d_surf_array[s]['d_zc'] = gpuarray.to_gpu(numpy.ravel(surf_array[s]['zc_sort']))
#        d_surf_array[s].xcDev = gpuarray.to_gpu(numpy.ravel(surf_array[
#              s].xcSort.astype(REAL)))
#        d_surf_array[s].ycDev = gpuarray.to_gpu(numpy.ravel(surf_array[
#              s].ycSort.astype(REAL)))
#        d_surf_array[s].zcDev = gpuarray.to_gpu(numpy.ravel(surf_array[
#            s].zcSort.astype(REAL)))

        #       Avoid transferring size 1 arrays to GPU (some systems crash)
        Nbuff = 5
        if len(surf_array[s]['size_target']) < Nbuff:
            size_target_buffer = numpy.zeros(Nbuff, dtype=numpy.int32)
            size_target_buffer[:len(surf_array[s]['size_target'])] = surf_array[
                s]['size_target'][:]
            d_surf_array[s]['d_size_target'] = gpuarray.to_gpu(size_target_buffer)
        else:
            d_surf_array[s]['d_size_target'] = gpuarray.to_gpu(surf_array[
                s]['size_target'].astype(numpy.int32))

    #        surf_array[s].['d_size_target'] = gpuarray.to_gpu(surf_array[s].size_target.astype(numpy.int32))
        d_surf_array[s]['d_offset_source'] = gpuarray.to_gpu(surf_array[s]['offset_source'].astype(numpy.int32))
        d_surf_array[s]['d_offset_twig'] = gpuarray.to_gpu(numpy.ravel(surf_array[s]['offset_twigs'].astype(numpy.int32)))
        d_surf_array[s]['d_offset_mlt'] = gpuarray.to_gpu(numpy.ravel(surf_array[s]['offsetMlt'].astype(numpy.int32)))
        d_surf_array[s]['d_M2P_lst'] = gpuarray.to_gpu(numpy.ravel(surf_array[s]['M2P_list'].astype(numpy.int32)))
        d_surf_array[s]['d_P2P_lst'] = gpuarray.to_gpu(numpy.ravel(surf_array[s]['P2P_list'].astype(numpy.int32)))
        d_surf_array[s]['d_xk'] = gpuarray.to_gpu(surf_array[s]['xk'].astype(REAL))
        d_surf_array[s]['d_wk'] = gpuarray.to_gpu(surf_array[s]['wk'].astype(REAL))
        d_surf_array[s]['d_Xsk'] = gpuarray.to_gpu(surf_array[s]['Xsk'].astype(REAL))
        d_surf_array[s]['d_Wsk'] = gpuarray.to_gpu(surf_array[s]['Wsk'].astype(REAL))
        d_surf_array[s]['d_k'] = gpuarray.to_gpu((surf_array[s]['sort_source'] % param.K).astype(numpy.int32))

    ind.indexDev = gpuarray.to_gpu(ind.index_large.astype(numpy.int32))

    Nfield = len(field_array)
    for f in range(Nfield):
        if len(field_array[f].xq) > 0:
            field_array[f].xq_gpu = gpuarray.to_gpu(field_array[
                f].xq[:, 0].astype(REAL))
            field_array[f].yq_gpu = gpuarray.to_gpu(field_array[
                f].xq[:, 1].astype(REAL))
            field_array[f].zq_gpu = gpuarray.to_gpu(field_array[
                f].xq[:, 2].astype(REAL))
            field_array[f].q_gpu = gpuarray.to_gpu(field_array[f].q.astype(
                REAL))
