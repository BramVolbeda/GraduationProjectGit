import torch
import math
from PINN_Bram.scripts.NNbuilder import Net2, Net2P, init_normal, NNreader
import matplotlib.pyplot as plt
from PINN_Bram.scripts.fileReaders import fileReader, dataReader
import numpy as np

nPt = 200
X_scale = 3.0
Y_scale = 2.0
device = 'cuda'
directory = "C:/Users/s163213/OneDrive/Graduation project/Code/PINN/Data/2D-aneurysm/"
vel_file = directory + "velocity_IA_steady.vtu"
x, y, z, _ = fileReader(vel_file, 2, mesh=True)
x = x / X_scale
y = y / Y_scale
x = torch.tensor(x).to(device)
y = torch.tensor(y).to(device)
x = x.type(torch.cuda.FloatTensor)
y = y.type(torch.cuda.FloatTensor)

M = torch.tensor([1.8, 0.7, 0])
radius = 0.5
x_circle = torch.zeros(nPt)
y_circle = torch.zeros(nPt)
x_inner_c = torch.zeros(nPt)
y_inner_c = torch.zeros(nPt)
radians = torch.linspace(-0.60, 3.70, nPt)
perc = 0.95

net2_u = Net2(2, 128).to(device)
net2_v = Net2(2, 128).to(device)
net2_p = Net2P(2, 128).to(device)

path = "C:/Users\s163213\OneDrive\Graduation project\Code\Results/2Daneurysm/"

print('Reading (pretrain) functions first...')
net2_u.load_state_dict(torch.load(path + "/NNfiles/" + "IA_data_u_e5500" + ".pt")) # nog niet echt 5500
net2_v.load_state_dict(torch.load(path + "/NNfiles/" + "IA_data_v_e5500" + ".pt"))
net2_p.load_state_dict(torch.load(path + "/NNfiles/" + "IA_data_p_e5500" + ".pt")) #moeite met druk volgens mij

M_np = np.array([[1.8], [0.7], [0]])
radius_np = 0.5
theta = np.linspace(-0.60, 3.70, nPt)
circle = np.array([radius_np*np.cos(theta) + M_np[0], radius_np*np.sin(theta)+M_np[1]])

e_n = np.zeros((nPt,2))
len_e = np.zeros(nPt)


for i in range(nPt):
    e_n_i = np.array([circle[0,i] - M_np[0], circle[1,i] - M_np[1]])
    mag = np.sqrt(e_n_i[0]**2 + e_n_i[1]**2)
    e_n[i, 0] = (circle[0, i] - M_np[0]) / mag * -1
    e_n[i,1] = (circle[1,i] - M_np[1]) / mag * -1
    len_e[i] = np.sqrt(e_n[i,0] **2 + e_n[i,1]**2)


x_circ = circle[0,:]
x_circ = x_circ.reshape(-1, 1)
y_circ = circle[1,:]
y_circ = y_circ.reshape(-1, 1)
e_nx = e_n[:,0]
e_nx = e_nx.reshape(-1, 1)
e_ny = e_n[:,1]
e_ny = e_ny.reshape(-1, 1)

plt.figure()
fig, ax = plt.subplots(figsize=(9,9))
skip=(slice(None,None,5),slice(None,None,5)) #plot every 5 pts
#ax.quiver(x.detach().numpy(), y.detach().numpy(), output_u , output_v,scale=5)
ax.quiver(x_circ[skip], y_circ[skip], e_nx[skip], e_ny[skip],scale=50)#a smaller scale parameter makes the arrow longer.
# ax[1].quiver(x_inner_e[skip], y_inner_e[skip], e_nX2[skip], e_nY2[skip],scale=50)
# ax.quiver(x.detach().numpy()[skip], y.detach().numpy()[skip], U[skip], V[skip],scale=50)
plt.title('NN results, Vel vector')
ax.axis('equal')
# plt.show()


x_circle = torch.tensor(x_circ)
y_circle = torch.tensor(y_circ)
x_circle = x_circle.type(torch.cuda.FloatTensor)
y_circle = y_circle.type(torch.cuda.FloatTensor)

y_circle = y_circle.view(nPt, -1)  # vs len(y_data) in 2D stenose
x_circle = x_circle.view(nPt, -1)
# y_inner_c = y_inner_c.view(nPt, -1)
# x_inner_c = x_inner_c.view(nPt, -1)

# x_inner_e = x_inner_c / X_scale
# y_inner_e = y_inner_c / Y_scale

x_inner_e = x_circle / X_scale
y_inner_e = y_circle / Y_scale

x_inner_e = x_inner_e.to(device)
y_inner_e = y_inner_e.to(device)
x_inner_e.requires_grad = True
y_inner_e.requires_grad = True
x.requires_grad = True
y.requires_grad = True

net_in = torch.cat((x_inner_e, y_inner_e), 1) #inner ellips coordinates are put into NN
# net_in = torch.cat((x, y), 1)

u = net2_u(net_in)
u = u.view(len(u), -1) # rearrange the tensor, geef het len(u) rijen, grootte column (-1) mag het zelf uitrekenen
v = net2_v(net_in)
v = v.view(len(v), -1)  # (256, 1 ) afhankelijk dus van batch_size grootte
P = net2_p(net_in)
P = P.view(len(P), -1)

u_y = torch.autograd.grad(u, y_inner_e, grad_outputs=torch.ones_like(y_inner_e), create_graph=True, only_inputs=True)[0]
u_x = torch.autograd.grad(u, x_inner_e, grad_outputs=torch.ones_like(x_inner_e), create_graph=True, only_inputs=True)[0]

v_y = torch.autograd.grad(v, y_inner_e, grad_outputs=torch.ones_like(y_inner_e), create_graph=True, only_inputs=True)[0]
v_x = torch.autograd.grad(v, x_inner_e, grad_outputs=torch.ones_like(x_inner_e), create_graph=True, only_inputs=True)[0]

# u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
# u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
#
# v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
# v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]

mu = 4e-3  # https://www.engineeringtoolbox.com/absolute-viscosity-liquids-d_1259.html

stress = torch.zeros(nPt, 2, 2)  # 200 points, 3x3 matrix for each point
stress_n = torch.zeros(nPt, 2)
stress_nn = torch.zeros(nPt, 1).to(device)
stress_nn_vec = torch.zeros(nPt, 2)
WSS_vec = torch.zeros(nPt, 2)
WSS_mag = torch.zeros(nPt, 1)

e_nx = torch.tensor(e_nx).to(device)
e_ny = torch.tensor(e_ny).to(device)
# e_nx = e_nx.cpu()
# e_ny = e_ny.cpu()

for i in range(nPt):

    # building stress tensor
    p = P[i]  # local pressure
    ux_i = u_x[i]
    uy_i = u_y[i]
    vx_i = v_x[i]
    vy_i = v_y[i]
    stress[i, 0, 0] = 2*ux_i - p
    stress[i, 0, 1] = uy_i + vx_i

    stress[i, 1, 0] = uy_i + vx_i
    stress[i, 1, 1] = 2*vy_i - p

    #niet vergeten stress aan het eind met mu te vermenigvuldigen

    stress_n[i, 0] = (2 * ux_i - p) * e_nx[i] + ((uy_i + vx_i) * e_ny[i])   # rij komt overeen met index punt
    stress_n[i, 1] = (2 * uy_i - p) * e_ny[i] + ((uy_i + vx_i) * e_nx[i])

    stress_nn[i] = e_nx[i] * stress_n[i, 0] + e_ny[i] * stress_n[i, 1]   #  make it a vector
    stress_nn_vec[i, 0] = stress_nn[i] * e_nx[i]
    stress_nn_vec[i, 1] = stress_nn[i] * e_ny[i]

    WSS_vec[i, :] = torch.mul(mu, stress_n[i, :]) - torch.mul(mu, stress_nn_vec[i, :])  # WSS vector
    WSS_mag[i] = math.sqrt(torch.inner(WSS_vec[i, :], WSS_vec[i, :]))

WSS_max = torch.max(WSS_mag).item()
WSS_norm = torch.div(WSS_mag, WSS_max)

X_max_c = torch.max(x_circle).item()
X_min_c = torch.min(x_circle).item()
# radius_c = (X_max_c - X_min_c) / 2  # radius of the circle

frac_c = (radians[-1] - radians[0])/(2*np.pi)  # fractie van cirkel meegenomen voor WSS
S = torch.linspace(0, radius*frac_c*2*np.pi, nPt)  # 'parametrization', circumferention


plt.figure()
plt.plot(S, WSS_norm)
plt.xlabel('S')
plt.ylabel('WSS/WSSmax')
plt.title('WSS/WSSmax versus circumferention S')
plt.show()

u = u.cpu()
v = v.cpu()
x = x.cpu()
y = y.cpu()
x_inner_e = x_inner_e.cpu()
y_inner_e = y_inner_e.cpu()
u = u.detach().numpy()
v = v.detach().numpy()

U = u / np.sqrt(u**2 + v**2)
V = v / np.sqrt(u**2 + v **2)

plt.figure()
fig, ax = plt.subplots(figsize=(9,9))
skip=(slice(None,None,5),slice(None,None,5)) #plot every 5 pts
#ax.quiver(x.detach().numpy(), y.detach().numpy(), output_u , output_v,scale=5)
# ax.quiver(x_inner_e.detach().numpy()[skip], y_inner_e.detach().numpy()[skip], e_nx[skip], e_ny[skip],scale=50)#a smaller scale parameter makes the arrow longer.
# ax[1].quiver(x_inner_e[skip], y_inner_e[skip], e_nX2[skip], e_nY2[skip],scale=50)
ax.quiver(x.detach().numpy()[skip], y.detach().numpy()[skip], U[skip], V[skip],scale=50)
plt.title('NN results, Vel vector')
plt.show()

# plt.subplot(211)
# ax1 = plt.subplot(111)
# ax2 = plt.subplot(211)
# skip=(slice(None,None,5),slice(None,None,5)) #plot every 5 pts
# #ax.quiver(x.detach().numpy(), y.detach().numpy(), output_u , output_v,scale=5)
# ax1.quiver(x_inner_e.detach().numpy()[skip], y_inner_e.detach().numpy()[skip], e_nX[skip], e_nY[skip],scale=50)#a smaller scale parameter makes the arrow longer.
# ax2.quiver(x_inner_e[skip], y_inner_e[skip], e_nX2[skip], e_nY2[skip],scale=50)
# # ax.quiver(x.detach().numpy()[skip], y.detach().numpy()[skip], U[skip], V[skip],scale=50)
# plt.title('NN results, Vel vector')
# plt.show()
