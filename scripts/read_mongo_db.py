import matplotlib.pyplot as plt
import pymongo
import io
import gridfs
import numpy as np
import pandas as pd


auth_url = 'mongodb+srv://dbBram:t42F7C828K!!@experiments.aakys.mongodb.net/test_mongo?ssl=true&ssl_cert_reqs=CERT_NONE'
client = pymongo.MongoClient(auth_url)

def cout(_id):
    df = io.StringIO(r.find({'_id': _id})[0]['captured_out']).read()
    return df

def experiment_name(_id):
    df = r.find({'_id': _id})[0]['experiment']['name']
    return df

def runs():
    return db.get_collection('runs')

def metrics():
    m = db.get_collection('metrics')
    return m

def run(_id):
    return r.find({'_id': _id})[0]

def metric(m,run_id,loss_nr):
    return m.find({'run_id' : run_id})[loss_nr]
    # return r.find({'run_id}': run_id})[0] # misschien meer dan alleen eerste instantie

def config(_id):
    df = run(_id)['config']
    return df


def source(_id, print_files=False): # retrieve the .py source file if needed
    fs = gridfs.GridFS(db) #mongodb function
    ### load keras model from json:
    files = db['fs.files']
    r_ = list(r.find({"_id": _id}))[0]
    source_files = r_['experiment']['sources']
    files = {}
    for sf in source_files:
        file_id = sf[-1]
        s = fs.get(file_id).read()
        files[sf[0]] = s.decode()
        if print_files:
            print(files[sf[0]])
            print('\n\n\n')
    return files

def collect_run_output(start_id, end_id, experiment_name=None, skip_ids=[], verbose=10):
    # get an overview of all the experiments in the database and their exit status
    output = []
    names = []
    for i in range(start_id, end_id + 1):
        if any(i == x for x in skip_ids):
            print('Run {} skipped'.format(i))
            continue
        try:
            run_dict = run(i)
        except IndexError as e:
            if verbose >= 5:
                print('No experiment exists with id {}'.format(i))
                print('Skipping next ids...')
                break
        try:
            result = run_dict['result']
        except KeyError as e:
            result = None
        if result is None:
            if verbose >= 10:
                print('Run {} did not finish'.format(i))
            continue

        name = run_dict['experiment']['name']
        if experiment_name is not None and experiment_name != name:
            if verbose >= 5:
                print('Run {} skipped since experiment name does not match set parameter: {}'.format(i, name))
            continue
        names.append(name)

        output.append(run_dict)

    uni_names = np.unique(names, return_counts=True)
    if len(uni_names[0]) == 1:
        if verbose >= 1:
            print('All these experiments belong to experiment: {}'.format(uni_names[0][0]))
    else:
        print('WARNING: there are multiple experiments present in given range')
        print('It is advised to set the experiment_name parameter to filter just one')
        print(uni_names)

    return output

def collect_metric_output(m,nr_losses, run_id, experiment_name=None, verbose=10):
    # get an overview of all the experiments in the database and their exit status
    #verbose >= 10 : notify if ID is not known, notify if a run did not finish (empty losses), print experiment name
    #verbose >= 5 : notify if ID is not known, print experiment name
    #verbose >= 1 : print experiment name
    #verbose = 0 : no print output
    steps = []
    output = []
    names = []
    loss_list = [[] for _ in range(nr_losses)]

    for j in range(nr_losses):
        try:
            run_dict = metric(m,run_id,j) # m == database, i = run_idx, j = loss_idx
        except IndexError as e:
            if verbose >= 5:
                print('No experiment exists with id {}'.format(run_id))
                print('Skipping next ids...')
                break
        try:
            loss_list[j] = run_dict['values']
            print(run_dict['name'])
        except KeyError as e:
            loss_list[j] = None

        if loss_list[j] is None:
            if verbose >= 10:
                print('Run {} did not finish'.format(run_id))
            continue

        name = run_dict['name']
        if experiment_name is not None and experiment_name != name:
            if verbose >= 5:
                print('Run {} skipped since experiment name does not match set parameter: {}'.format(run_id, name))
            continue
        names.append(name)

        output.append(run_dict)

    steps = run_dict['steps']
    uni_names = np.unique(names, return_counts=True)
    return loss_list, steps

db = client['STEN2DA_']
case_name = 'stenose2DA'

nr_losses = 3 # number of different loss functions - 15 for latest runs 
PINN_losses = 3
run_id = 5
run_id2 = 9
run_id3 = 6

std_x = 0.6
std_y = 0.08
std_x2 = 0.3
std_y2 = 0.08 

flag_compare = False
epoch_save_loss = 2 
epoch_save_ns = 10
comparing = ['MSE', 'MAE']  # Only first index matters if nothing is being compared

r = runs()
m = metrics()
# get the .py file

file = source(run_id, print_files=False)
experiment = run(run_id)
losses, steps = collect_metric_output(m, nr_losses=nr_losses, run_id=run_id)
colors = ['b', 'r', 'g']
steps_loss = np.linspace(0, int(np.ceil(len(losses[0]) * epoch_save_loss - epoch_save_loss)), int(len(losses[0])))

if flag_compare:
    
    experiment2 = run(run_id2)
    losses2, steps2 = collect_metric_output(m, nr_losses=nr_losses, run_id=run_id2)
    colors2 = ['turquoise', 'limegreen', 'lightsalmon']
    steps_loss2 = np.linspace(0, int(np.ceil(len(losses2[0]) * epoch_save_loss - epoch_save_loss)), int(len(losses2[0])))

    # experiment3 = run(run_id3)
    # losses3, steps3 = collect_metric_output(m, nr_losses=nr_losses, run_id=run_id3)
    # colors3 = ['darkviolet', 'greenyellow', 'sandybrown']
    # steps_loss3 = np.linspace(0, int(np.ceil(len(losses3[0]) * epoch_save_loss - epoch_save_loss)), int(len(losses3[0])))

# experiment_metrics = metric(m,7,1) # give run_id

if flag_compare: 
    title_list = ['eq_loss', 'bc_loss', 'data_loss']
    # ylim_list = [[0.01, 0.4],[1e-7, 0.001], [3e-6, 0.001]]
    fig = plt.figure()
    for i in range(PINN_losses):
        ax = fig.add_subplot(1, 3, i+1)
        plt.plot(steps_loss, losses[i], color='b')  # colors[i])
        plt.plot(steps_loss2, losses2[i], color='limegreen')  # colors2[i])
        # plt.plot(steps_loss3, losses3[i], color='r')  # colors3[i])
        ax.set_yscale('log')
        plt.xlabel('epochs')
        plt.ylabel('Loss')
        plt.title(title_list[i])
        plt.legend(comparing)
        # plt.ylim(ylim_list[i])
    plt.suptitle(case_name)
    plt.show()

else: 
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i in range(PINN_losses):
        plt.plot(steps_loss, losses[i], colors[i])
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    ax.set_yscale('log')
    plt.legend(['eq', 'bc', 'data'])
    plt.title(case_name) 
    plt.show()

df1 = pd.DataFrame(losses).T
steps_ns = np.linspace(0, int(np.ceil(len(losses[0])*epoch_save_ns - epoch_save_ns)), int(len(losses[0])))
steps_ns2 = np.linspace(0, int(np.ceil(len(losses2[0])*epoch_save_ns - epoch_save_ns)), int(len(losses2[0])))
# steps_ns3 = np.linspace(0, int(np.ceil(len(losses3[0])*epoch_save_ns - epoch_save_ns)), int(len(losses3[0])))
title_list_ns = ['convX', 'convY', 'diffX', 'diffY', 'pressure', 'gradients']
# color_list = ['b', 'turquoise', 'r', 'lightsalmon'] 
color_list = ['b', 'turquoise', 'g', 'limegreen', 'r', 'lightsalmon']
# nu * d^2 en 1/rho*dp 
# legend_list = [['U\u00B7\u2202u/\u2202x' + " " + str(comparing[0]), 'V\u00B7\u2202u/\u2202y' + " " + str(comparing[0])], 
#                 ['U\u00B7\u2202v/\u2202x' + " " + str(comparing[0]), 'V\u00B7\u2202v/\u2202y' + " " + str(comparing[0])],
#                 ['\u03BD\u00B7\u2202$^2$u/\u2202x$^2$' + " " + str(comparing[0]), '\u03BD\u00B7\u2202$^2$u/\u2202y$^2$' + " " + str(comparing[0])],
#                 ['\u03BD\u00B7\u2202$^2$v/\u2202x$^2$' + " " + str(comparing[0]), '\u03BD\u00B7\u2202$^2$v/\u2202y$^2$' + " " + str(comparing[0])],
#                 ['1/\u03C1\u00B7\u2202P/\u2202x' + " " + str(comparing[0]), '1/\u03C1\u00B7\u2202P/\u2202y' + " " + str(comparing[0])], 
#                     ['U\u00B7\u2202u/\u2202x' + " " + str(comparing[1]), 'V\u00B7\u2202u/\u2202y' + " " + str(comparing[1])], 
#                     ['U\u00B7\u2202v/\u2202x' + " " + str(comparing[1]), 'V\u00B7\u2202v/\u2202y' + " " + str(comparing[1])], 
#                     ['\u03BD\u00B7\u2202$^2$u/\u2202x$^2$' + " " + str(comparing[1]), '\u03BD\u00B7\u2202$^2$u/\u2202y$^2$' + " " + str(comparing[1])], 
#                     ['\u03BD\u00B7\u2202$^2$v/\u2202x$^2$' + " " + str(comparing[1]), '\u03BD\u00B7\u2202$^2$v/\u2202y$^2$' + " " + str(comparing[1])], 
#                     ['1/\u03C1\u00B7\u2202P/\u2202x' + " " + str(comparing[1]), '1/\u03C1\u00B7\u2202P/\u2202y' + " " + str(comparing[1])]]
# d^2 en dp
if nr_losses == 13: 
    loss_X = 4
    legend_list = [['U\u00B7\u2202u/\u2202x' + " " + str(comparing[0]), 'V\u00B7\u2202u/\u2202y' + " " + str(comparing[0])], 
                    ['U\u00B7\u2202v/\u2202x' + " " + str(comparing[0]), 'V\u00B7\u2202v/\u2202y' + " " + str(comparing[0])],
                    ['\u2202$^2$u/\u2202x$^2$' + " " + str(comparing[0]), '\u2202$^2$u/\u2202y$^2$' + " " + str(comparing[0])],
                    ['\u2202$^2$v/\u2202x$^2$' + " " + str(comparing[0]), '\u2202$^2$v/\u2202y$^2$' + " " + str(comparing[0])],
                    ['\u2202P/\u2202x' + " " + str(comparing[0]), '\u2202P/\u2202y' + " " + str(comparing[0])], 
                        ['U\u00B7\u2202u/\u2202x' + " " + str(comparing[1]), 'V\u00B7\u2202u/\u2202y' + " " + str(comparing[1])], 
                        ['U\u00B7\u2202v/\u2202x' + " " + str(comparing[1]), 'V\u00B7\u2202v/\u2202y' + " " + str(comparing[1])], 
                        ['\u2202$^2$u/\u2202x$^2$' + " " + str(comparing[1]), '\u2202$^2$u/\u2202y$^2$' + " " + str(comparing[1])], 
                        ['\u2202$^2$v/\u2202x$^2$' + " " + str(comparing[1]), '\u2202$^2$v/\u2202y$^2$' + " " + str(comparing[1])], 
                        ['\u2202P/\u2202x' + " " + str(comparing[1]), '\u2202P/\u2202y' + " " + str(comparing[1])]]
else: 
    loss_X = 5
    legend_list = [['U\u00B7\u2202u/\u2202x' + " " + str(comparing[0]), 'V\u00B7\u2202u/\u2202y' + " " + str(comparing[0])], 
                    ['U\u00B7\u2202v/\u2202x' + " " + str(comparing[0]), 'V\u00B7\u2202v/\u2202y' + " " + str(comparing[0])],
                    ['\u2202$^2$u/\u2202x$^2$' + " " + str(comparing[0]), '\u2202$^2$u/\u2202y$^2$' + " " + str(comparing[0])],
                    ['\u2202$^2$v/\u2202x$^2$' + " " + str(comparing[0]), '\u2202$^2$v/\u2202y$^2$' + " " + str(comparing[0])],
                    ['\u2202P/\u2202x' + " " + str(comparing[0]), '\u2202P/\u2202y' + " " + str(comparing[0])],
                    ['\u2202u/\u2202x' + " " + str(comparing[0]), '\u2202v/\u2202y' + " " + str(comparing[0])],  
                        ['U\u00B7\u2202u/\u2202x' + " " + str(comparing[1]), 'V\u00B7\u2202u/\u2202y' + " " + str(comparing[1])], 
                        ['U\u00B7\u2202v/\u2202x' + " " + str(comparing[1]), 'V\u00B7\u2202v/\u2202y' + " " + str(comparing[1])], 
                        ['\u2202$^2$u/\u2202x$^2$' + " " + str(comparing[1]), '\u2202$^2$u/\u2202y$^2$' + " " + str(comparing[1])], 
                        ['\u2202$^2$v/\u2202x$^2$' + " " + str(comparing[1]), '\u2202$^2$v/\u2202y$^2$' + " " + str(comparing[1])], 
                        ['\u2202P/\u2202x' + " " + str(comparing[1]), '\u2202P/\u2202y' + " " + str(comparing[1])],
                        ['\u2202u/\u2202x' + " " + str(comparing[1]), '\u2202v/\u2202y' + " " + str(comparing[1])], 
                        ['U\u00B7\u2202u/\u2202x' + " " + str(comparing[2]), 'V\u00B7\u2202u/\u2202y' + " " + str(comparing[2])], 
                        ['U\u00B7\u2202v/\u2202x' + " " + str(comparing[2]), 'V\u00B7\u2202v/\u2202y' + " " + str(comparing[2])], 
                        ['\u2202$^2$u/\u2202x$^2$' + " " + str(comparing[2]), '\u2202$^2$u/\u2202y$^2$' + " " + str(comparing[2])], 
                        ['\u2202$^2$v/\u2202x$^2$' + " " + str(comparing[2]), '\u2202$^2$v/\u2202y$^2$' + " " + str(comparing[2])], 
                        ['\u2202P/\u2202x' + " " + str(comparing[2]), '\u2202P/\u2202y' + " " + str(comparing[2])],
                        ['\u2202u/\u2202x' + " " + str(comparing[2]), '\u2202v/\u2202y' + " " + str(comparing[2])]]

if flag_compare:    
    df2 = pd.DataFrame(losses2).T
    # df3 = pd.DataFrame(losses3).T

fig2 = plt.figure()
j = 1
for i in range(PINN_losses, nr_losses, 2):
    ax = fig2.add_subplot(2, 3, j)

    plt.plot(steps_ns, df1.iloc[:, i].rolling(50, min_periods=1).mean(), color_list[0])
    plt.plot(steps_ns, df1.iloc[:, i+1].rolling(50, min_periods=1).mean(), color_list[1])
    if flag_compare: 
        plt.plot(steps_ns2, df2.iloc[:, i].rolling(50, min_periods=1).mean(), color_list[2])
        plt.plot(steps_ns2, df2.iloc[:, i+1].rolling(50, min_periods=1).mean(), color_list[3])
        # plt.plot(steps_ns3, df3.iloc[:, i].rolling(50, min_periods=1).mean(), color_list[4])
        # plt.plot(steps_ns3, df3.iloc[:, i+1].rolling(50, min_periods=1).mean(), color_list[5])
    plt.title(title_list_ns[j-1])
    if j == 1: 
        plt.legend(['MSE', 'MSE', 'Hubers', 'Hubers', 'MAE', 'MAE'])
    # plt.legend(legend_list[j-1] + legend_list[j+loss_X]+ legend_list[j+2*loss_X-1]) if flag_compare else plt.legend(legend_list[j-1])
    # plt.xlabel('epochs')
    plt.ylabel('Loss')
    ax.set_yscale('log')
    # plt.ylim(ylim_list2[j-1])
    j += 1
    plt.suptitle('NS eq. loss components for {} vs {}, {}'.format(comparing[0], comparing[1], case_name)) if flag_compare \
        else plt.suptitle('NS eq. loss components for {}, {}'.format(comparing[0], case_name))
plt.show()

# fig3 = plt.figure()
# ax = fig3.add_subplot(1, 1, 1)
# plt.plot(steps_loss, df1.iloc[:, -2].rolling(50, min_periods=1).mean(), color_list[0])
# plt.plot(steps_loss, df1.iloc[:, -1].rolling(50, min_periods=1).mean(), color_list[2])
# if flag_compare: 
#     plt.plot(steps_loss2, df2.iloc[:, -2].rolling(50, min_periods=1).mean(), color_list[1])
#     plt.plot(steps_loss2, df2.iloc[:, -1].rolling(50, min_periods=1).mean(), color_list[3])
# plt.title(title_list_ns[-1])
# plt.legend(legend_list[-1])
# plt.xlabel('epochs')
# plt.ylabel('Loss')
# ax.set_yscale('log')
# plt.suptitle('NS eq. loss components for {} vs {}, {}'.format(comparing[0], comparing[1], case_name))
# plt.show()


fig2 = plt.figure()
j = 1
for i in range(PINN_losses, nr_losses, 2):
    ax = fig2.add_subplot(2, 3, j)

    plt.plot(steps_ns, df1.iloc[:, i], color_list[0])
    plt.plot(steps_ns, df1.iloc[:, i+1], color_list[2])
    if flag_compare: 
        plt.plot(steps_ns2, df2.iloc[:, i], color_list[1])
        plt.plot(steps_ns2, df2.iloc[:, i+1], color_list[3])
    
    plt.title(title_list_ns[j-1])
    plt.legend(legend_list[j-1] + legend_list[j+4]) if flag_compare else plt.legend(legend_list[j-1])
    # plt.xlabel('epochs')
    plt.ylabel('Loss')
    ax.set_yscale('log')
    # plt.ylim(ylim_list2[j-1])
    j += 1
    plt.suptitle('NS eq. loss components for {} vs {}, {}'.format(comparing[0], comparing[1], case_name)) if flag_compare \
        else plt.suptitle('NS eq. loss components for {}, {}'.format(comparing[0], case_name))
plt.show()




X_scale = 2.0
Y_scale = 1.0
XX_scale = 4.0
YY_scale = 1.0
UU_scale = 1.0

convU_ux = df2.iloc[:, 3]  # TODO is aangepast in loss_geov3 
convV_uy = df2.iloc[:, 4]
convU_vx = df2.iloc[:, 5]
convV_vy = df2.iloc[:, 6]
diffXu = df2.iloc[:, 7]
diffYu = df2.iloc[:, 8]
diffXv = df2.iloc[:, 9]
diffYv = df2.iloc[:, 10]
forceX = df2.iloc[:, 11]
forceY = df2.iloc[:, 12]

# loss_1 = convU_ux / X_scale + convV_uy / Y_scale - (diffXu / XX_scale + diffYu / YY_scale ) \
#                  + forceX / (X_scale * UU_scale)
# loss_2 = convU_vx / X_scale + convV_vy / Y_scale - (diffXv / XX_scale + diffYv / YY_scale ) \
#                  + forceY / (Y_scale * UU_scale)  # Y-dir

# loss_3 = (u_x / X_scale + v_y / Y_scale)  # continuity

# print(experiment)
# # id = experiment.get('epochs')
# print(experiment.get('training.loss_eqn'))
# output = collect_run_output(1, 10)
# # output[1]['result']
# df = experiment_name(3)
# print(df)

# const { MongoClient } = require("mongodb")
# # // Replace the uri string with your MongoDB deployment's connection string.
# const uri =
#   "mongodb+srv://<user>:<password>@<cluster-url>?writeConcern=majority";
# const client = new MongoClient(uri)
# async function run() {
#   try {
#     await client.connect()