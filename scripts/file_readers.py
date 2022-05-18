import numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
import os

def file_reader(file, mesh, **extremes):
    """
    Function that takes in a *.vtk/*.vtu file and extracts the coordinates. 
    Outputs minima and maxima for equal scaling between data files.  
    
    Parameters:
    file (*.vtk/*.vtu) : The file containing either the mesh or the boundary coordinates. 
    mesh (Boolean) : If the file is of type .*vtu (mesh) or *.vtk (boundary). 
    **extremes (kwargs) : Minima and maxima lists from previous function call. Defined as min and max. 

    Returns: 
    vector_norm (list) : Coordinates normalized between [-1, 1], for x (index 0), y (index 1) and z (index 2). 
    minima (list) : Minumum values for x (index 0), y (index 1) and z (index 2). 
    maxima (list) : Maximum values for x (index 0), y (index 1) and z (index 2). 
    """
    print('Loading', file)  # TODO f string format 
    # reader = vtk.vtkXMLUnstructuredGridReader() if mesh else vtk.vtkUnstructuredGridReader() # Set up the reader type 
    reader = vtk.vtkXMLUnstructuredGridReader() if mesh else vtk.vtkPolyDataReader() # Set up the reader type 
    # elif input_n == 3:
    #     reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file) 
    reader.Update() 
    data_vtk = reader.GetOutput()
    n_points = data_vtk.GetNumberOfPoints() 
    print('n_points of the mesh:', n_points)

    x_vtk_mesh = np.zeros((n_points, 1))
    y_vtk_mesh = np.zeros((n_points, 1))
    z_vtk_mesh = np.zeros((n_points, 1))
    
    VTKpoints = vtk.vtkPoints()

    for i in range(n_points):   # Read the contents of the file
        pt_iso = data_vtk.GetPoint(i)
        x_vtk_mesh[i] = '%.4f'%(pt_iso[0])  # Expressed in 4 decimals, avoid close to 0 truncation errors
        y_vtk_mesh[i] = '%.4f'%(pt_iso[1])
        z_vtk_mesh[i] = '%.4f'%(pt_iso[2])
        VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2]) 

    point_data = vtk.vtkUnstructuredGrid() 
    point_data.SetPoints(VTKpoints) 


    # Setting up the location vector, division error if Z only contains zeros
    location_vector = [x_vtk_mesh, y_vtk_mesh, z_vtk_mesh] if np.any(z_vtk_mesh) else [x_vtk_mesh, y_vtk_mesh]
    lv_reshape = [vector.reshape((np.size(vector[:]), 1)) for vector in location_vector]
    lv_transpose = [vector.reshape(-1, 1) for vector in lv_reshape]

    # Applying 0-mean normalization to the input data 
    means = [vector.mean() for vector in lv_transpose] 
    stds = [vector.std() for vector in lv_transpose]
    vector_norm_mean = [(vector - mean) / std for vector, mean, std in zip(lv_transpose, means, stds)]
    # append Z-dimension if lost earlier
    vector_norm_mean.append(np.zeros((len(vector_norm_mean[0]), 1))) if not np.any(z_vtk_mesh) else vector_norm_mean
    
     # Applying [-1, 1] normalization to the input data
    if len(extremes) != 0: 
        minima = extremes['min']
        maxima = extremes['max']
    else: 
        minima = [vector.min() for vector in lv_transpose]
        maxima = [vector.max() for vector in lv_transpose]
    vector_norm_range = [2.* (vector - minimum) / (maximum - minimum) -1 for vector, minimum, maximum in zip(lv_transpose, minima, maxima)]
    # append Z-dimension if lost earlier 
    vector_norm_range.append(np.zeros((len(vector_norm_range[0]), 1))) if not np.any(z_vtk_mesh) else vector_norm_range

    return vector_norm_range, minima, maxima


def data_reader(case, random_flag=False, **extremes):
    """
    Function that takes in the solution file (*.vtu) and extracts the velocity values at prespecified
    or random locations. 

    Parameters: 
    case (Class) : Corresponds to the class that contains information on the solution file used and the data
                    point locations
    random_flag (Boolean) : If set to True, the user is required to specify the amount of data points that will
                            be randomly sampled within the domain. If set to False, the pre-specified points will
                            be used as described in the case information. 
    **extremes (kwargs) : Minima and maxima lists from previous function call. Defined as min and max. 

    Returns: 
    solution_locations (list) : List of lists corresponding to the coordinates of the data points. 
                                Index 0 is x, index 1 is y, index 2 is z.  
    solution_values (list) : List of lists corresponding to the values of the data points. 
                                Index 0 is u, index 1 is v, index 2 is w. 

    """

    x_data = np.asarray(case.x_data) 
    y_data = np.asarray(case.y_data) 
    z_data = np.asarray(case.z_data)
    data_locations = [x_data, y_data, z_data]
    print('Loading', case.vel_file) 

    reader = vtk.vtkXMLUnstructuredGridReader()  # Set up reader
    # reader = vtk.vtkUnstructuredGridReader()  # Set up reader
    reader.SetFileName(case.vel_file)
    reader.Update()
    data_vtk = reader.GetOutput()
    n_points = data_vtk.GetNumberOfPoints() 
    print('n_points of the data file read:', n_points)

    if random_flag:  
        N_sample = int(input("Please specify the amount of data points randomly sampled from the geometry"))
        sample_idx = n_points / N_sample
        print('Sampling', N_sample, 'points from a total of', n_points, 'points')
        x_vtk_mesh = np.zeros((n_points, 1))
        y_vtk_mesh = np.zeros((n_points, 1))
        z_vtk_mesh = np.zeros((n_points, 1))

        location_vector = [x_vtk_mesh, y_vtk_mesh, z_vtk_mesh]
        VTKpoints = vtk.vtkPoints()
        N_pts_data = 0
        for i in range(n_points):
            if i % sample_idx == 0:
                pt_iso = data_vtk.GetPoint(i)
                x_vtk_mesh[N_pts_data] = pt_iso[0]
                y_vtk_mesh[N_pts_data] = pt_iso[1]
                z_vtk_mesh[N_pts_data] = pt_iso[2]
                N_pts_data += 1
        print('n_points sampled', N_pts_data)
        data_locations = [mesh[0:N_pts_data, 0] for mesh in location_vector]


    VTKpoints = vtk.vtkPoints()
    for i in range(len(data_locations[0])): 
        VTKpoints.InsertPoint(i, [axis[i] for axis in data_locations]) 

    point_data = vtk.vtkUnstructuredGrid()
    point_data.SetPoints(VTKpoints)

    probe = vtk.vtkProbeFilter()  # Sample data from specific locations
    probe.SetInputData(point_data)
    probe.SetSourceData(data_vtk)
    probe.Update()
    array = probe.GetOutput().GetPointData().GetArray(case.fieldname) 
    data_vel = vtk_to_numpy(array) 

    solution_values = [data_vel[:,i] for i in range(len(data_locations))] 

    # Setting up the location vector, division error if Z only contains zeros
    sv = data_locations[0:-1] if not np.any(z_data) else data_locations
    sv_reshape = [vector.reshape((np.size(vector[:]), 1)) for vector in sv]
    sv_transpose = [vector.reshape(-1, 1) for vector in sv_reshape]

    # Applying [-1, 1] normalization to the input data  TODO min en max vanuit mesh, niet datapunten
    if len(extremes) != 0: 
        minima = extremes['min']
        maxima = extremes['max']
    else: 
        minima = [vector.min() for vector in sv_transpose]
        maxima = [vector.max() for vector in sv_transpose]

    solution_locations = [2.* (vector - minimum) / (maximum - minimum) -1 for vector, minimum, maximum in zip(sv_transpose, minima, maxima)]
    # append Z-dimension if lost earlier 
    solution_locations.append(np.zeros((len(solution_locations[0]), 1))) if not np.any(z_data) else solution_locations

    
    # ud = data_vel_u / case.U_scale  # TODO how to scale U? 
    # vd = data_vel_v / case.U_scale
    # wd = data_vel_w / case.U_scale
    
    solution_values = [velocity / case.U_scale for velocity in solution_values]

    return solution_locations, solution_values