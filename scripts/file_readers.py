import numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy
import os

def file_reader(file, mesh):
    """
    Function that takes in a *.vtk/*.vtu file and extracts the coordinates and values (optional).
    
    Parameters:
    file (*.vtk/*.vtu) : The file containing either the mesh, the boundary coordinates, or the solution data. 

    Returns: 


    """
    print('Loading', file)  # TODO f string format 
    
    if mesh:
        reader = vtk.vtkXMLUnstructuredGridReader()  # Set up the reader type 
    # elif input_n == 3:
    #     reader = vtk.vtkPolyDataReader()
    else:
        reader = vtk.vtkUnstructuredGridReader()  

    reader.SetFileName(file)  # Read the contents of the file
    reader.Update() 
    data_vtk = reader.GetOutput()
    n_points = data_vtk.GetNumberOfPoints() 
    print('n_points of the mesh:', n_points)

    x_vtk_mesh = np.zeros((n_points, 1))
    y_vtk_mesh = np.zeros((n_points, 1))
    z_vtk_mesh = np.zeros((n_points, 1))
    
    VTKpoints = vtk.vtkPoints() 
    for i in range(n_points): 
        pt_iso = data_vtk.GetPoint(i)
        x_vtk_mesh[i] = pt_iso[0] 
        y_vtk_mesh[i] = pt_iso[1] 
        z_vtk_mesh[i] = pt_iso[2]
        VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2]) 

    point_data = vtk.vtkUnstructuredGrid() 
    point_data.SetPoints(VTKpoints) 

    x = np.reshape(x_vtk_mesh, (np.size(x_vtk_mesh[:]), 1))  # Size (n_points, 1)
    y = np.reshape(y_vtk_mesh, (np.size(y_vtk_mesh[:]), 1))  
    z = np.reshape(z_vtk_mesh, (np.size(z_vtk_mesh[:]), 1))

    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    z = z.reshape(-1, 1)

    return x, y, z, data_vtk


def dataReader(case, random_flag=False):
    # !!specify pts location here:

    x_data = np.asarray(case.x_data)  # convert to numpy
    y_data = np.asarray(case.y_data)  # convert to numpy
    z_data = np.asarray(case.z_data)
    print(x_data, y_data)
    print('Loading', case.vel_file) # mesh met X_scale = 2, velocity_sten_steady (is dit dan al de oplossing van 2D stenose?)
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(case.vel_file)
    reader.Update()
    data_vtk = reader.GetOutput()
    n_points = data_vtk.GetNumberOfPoints() # 21140
    print('n_points of the data file read:', n_points)

    if random_flag:
        N_sample = int(input("Please specify the amount of data points randomly sampled from the geometry"))
        sample_idx = n_points / N_sample
        print('Sampling', N_sample, 'points from a total of', n_points, 'points')
        x_vtk_mesh = np.zeros((n_points, 1))
        y_vtk_mesh = np.zeros((n_points, 1))
        z_vtk_mesh = np.zeros((n_points, 1))
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
        x_data = np.zeros((N_pts_data, 1))
        y_data = np.zeros((N_pts_data, 1))
        z_data = np.zeros((N_pts_data, 1))

        x_data[:, 0] = x_vtk_mesh[0:N_pts_data, 0]
        y_data[:, 0] = y_vtk_mesh[0:N_pts_data, 0]
        z_data[:, 0] = z_vtk_mesh[0:N_pts_data, 0]


    VTKpoints = vtk.vtkPoints()
    for i in range(len(x_data)):
        VTKpoints.InsertPoint(i, x_data[i], y_data[i], z_data[i]) # anders dan voorheen; hier worden de arbitraire
        # gegenereerde datapunten ingevoerd, 'sensor'

    point_data = vtk.vtkUnstructuredGrid()
    point_data.SetPoints(VTKpoints) # bouw een unstructured grid voor deze random input punten

    probe = vtk.vtkProbeFilter() # sample data values at specified point locations --> sensor punten
    probe.SetInputData(point_data)
    probe.SetSourceData(data_vtk)
    probe.Update()
    array = probe.GetOutput().GetPointData().GetArray(case.fieldname) # vtkFloatArray size tuple (5,3)
    # data_vel = vtkToNumpy(array)
    data_vel = vtk_to_numpy(array) # [[0. 0. 0.]x 5]

    data_vel_u = data_vel[:, 0]
    data_vel_v = data_vel[:, 1]
    data_vel_w = data_vel[:, 2]
    
    x_data = x_data / case.X_scale
    y_data = y_data / case.Y_scale
    z_data = z_data / case.Z_scale
    ud = data_vel_u / case.U_scale
    vd = data_vel_v / case.U_scale
    wd = data_vel_w / case.U_scale
    
    
    xd = x_data.reshape(-1, 1)  # need to reshape to get 2D array # worden 5 losse lijsten
    yd = y_data.reshape(-1, 1)  # [[0.15],[0.07], etc.
    zd = z_data.reshape(-1, 1)
    ud = ud.reshape(-1, 1)  # need to reshape to get 2D array
    vd = vd.reshape(-1, 1)  # need to reshape to get 2D array
    wd = wd.reshape(-1, 1)

    return xd, yd, zd, ud, vd, wd