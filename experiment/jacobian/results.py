import pickle 
import torch
import matplotlib.pyplot as plt
import sklearn
import numpy as np


def get_pickled(experiments):
    dicts = {}
    for exprmnt in experiments:
        path = "../../local/jacobian/" + exprmnt
        get_pickle = open(path,"rb")
        res = pickle.load(get_pickle)
        dicts[exprmnt] = torch.stack(res).detach().cpu().numpy()
    return dicts

def draw_hists(results):
    for result in results.keys():
        path_to_save = "../../local/jacobian/" + str(result)+"_hist.png"
        plot = plt.hist(results[result], bins = 30)
        title = str(result).replace("_", " ")
        plt.title(title)
        plt.xlabel('Square of the determinant of (J)^T*(J)')
        plt.ylabel('counts')
        plt.show()
        plt.savefig(path_to_save)
        plt.clf()


experiments = ['nvae_linear_posterior',
               'nvae_linear_prior',
               'svae_linear_posterior',
               'svae_linear_prior']


results = get_pickled(experiments)

draw_hists(results)
#plot = plt.hist(result, bins = 30)
#plt.show()
#plt.savefig("../../local/hist_svae_linear_posterior/plot.jpg")

