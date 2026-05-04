#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 16:28:50 2025

@author: bob-van-sluijs
"""


# %%
from spline_models import LSTM_model_latent_decay_simple_fb_matO1,GRU_model_latent_decay_simple_fb_matO, GRU_model_latent_decay_simple_fb_mat_smooth_log,GRU_model_latent_decay_simple_fb_mat
#MZ_model_latent_decay_simple_fb_matsmooth,MZ_model_latent_decay_simple_fb_mat, MZ_model_latent_decay_simple_fb_mat_twostep, GRU_model_latent_decay_simple_fb_mat_twostep, GRU_model_latent_decay_simple_fb_mat, LinearRNN_model_latent_decay_simple_fb_matf, GRU_model_latent_decay_simple_fb_matf  ,LinearRNN_model_latent_decay_simple_fb_mat#, LSTM_model_latent_decay,GRU_model_latent_decay, GRU_model_latent_decay_simple #CNN_model_vectorized# LSTM_model_step, TCN_model_vectorized, TransformerStream_model_vectorized
from neural_spline import NeuralSpline
import torch, cProfile, pstats, sys

path  = 'Data parsed pruned'  
# profiler = cProfile.Profile()
# profiler.enable()

listset = [
    # (2000,0.008,150,300,4,True,True),
    # (2000,0.0018,768,350,1,True,True),
    # (2000,0.0008,128,150,3,True,True),
    # (2000,0.008,150,300,4,True,True),
    (2000,0.0018,400,100,2,True,True),
    # (2000,0.008,150,300,4,True,True),
    (2000,0.0018,768,350,1,True,True),
    (2000,0.0008,128,350,3,True,True),
    # (2000,0.008,150,300,4,True,True),
    (2000,0.0018,400,350,2,True,True),
    # (2000,0.008,150,300,4,True,True),
    (2000,0.0018,768,350,1,True,True),
    (2000,0.0008,128,350,3,True,True),
    # (2000,0.008,150,300,4,True,True),
    (2000,0.0018,400,350,2,True,True),
    # (2000,0.008,150,300,4,True,True),
    (2000,0.0018,768,350,1,True,True),
    (2000,0.0008,128,350,3,True,True),
    # (2000,0.008,150,300,4,True,True),
    (2000,0.0018,400,350,2,True,True),
    ]
     
for i in range(0,60,1):
        epoch, decay, hidden, batch, num_layers, rescale, normalize = listset[i]
        epoch = 250
        lr = 0.002
        model = NeuralSpline(path = path,rescale_inputs = False)
        model.train_val(epochs=epoch, 
                lr=lr, 
                hidden=hidden, 
                batch = batch,
                num_layers = num_layers,
                normalize  = normalize,decay = decay,
                save_path  = "models/GRU_model_latent_decay_simple_fb_matO{}.pt".format(str(i+1450)), 
                model_cls  = GRU_model_latent_decay_simple_fb_matO)
        model.plot_convergence(save_dir = '/home/bob-van-sluijs/Desktop/GRU_omat sqrt/plot {} Omat_LSTM/'.format(str(i+1450)))






# for i in range(0,1,1):
#      epoch, decay, hidden, batch, num_layers, rescale, normalize = listset[i]
#      model = NeuralSpline(path = path,rescale_inputs = rescale)
# # %%
#      out = model.MC_sample_windows_many_small_additions()
#      model.load_ensemble_from_folder('/home/bob-van-sluijs/Desktop/model convergence/', model_factory = GRU_model_latent_decay_simple_fb_matO)
#      model.simulate_ensemble2()

# # %%



























# listset = [
#     (2000,0.008,750,300,1,False,False),
#     (2000,0.0018,750,300,1,False,False),
#     (2000,0.0008,600,300,1,False,False),
#     (2000,0.004,600,300,1,False,False),
#     (2000,0.008,400,300,1,False,False),
#     (2000,0.0018,400,300,1,False,False),
#     (2000,0.0008,400,300,1,False,False),
#     (2000,0.004,400,300,1,False,False),
    # (2000,0.008,80,570,2,False,False),
    # (2000,0.005,200,557,2,False,False),
    # (2000,0.005,150,557,2,False,False),
    # (2000,0.005,150,557,2,False,False),
    # (2000,0.005,150,557,2,False,False),
    # (2000,0.005,150,557,2,False,False),
    # (2000,0.005,150,557,2,False,False),
    # (2000,0.005,150,557,2,False,False),
    # (2000,0.005,150,557,2,False,False),
    # (2000,0.005,150,557,2,False,False),
    # (2000,0.005,150,557,2,False,False),
    # (2000,0.005,150,557,2,False,False),
    # (2000,0.005,150,557,2,False,False),
    # (2000, 0.0005, 175, 300, 2, False, False),
    # (2000, 0.005, 175, 300, 2, False, False),
    # (2000, 0.0025, 175, 300, 1, False, False),
    # (2000, 0.005, 350, 300, 2, False, False),
    # (2000, 0.005, 350, 300, 1, False, False),
    # (2000, 0.0005, 350, 300, 1, False, False),
    # (2000, 0.005, 500, 300, 1, False, False),
    # (2000, 0.0025, 500, 300, 1, False, False),
    # (2000, 0.005, 500, 300, 1, False, False),
    # (2000, 0.0005, 300, 300, 1, False, False),
    # (2000, 0.005, 300, 300, 1, False, False),
    # (2000, 0.0025, 300, 300, 1, False, False),
    # (2000, 0.005, 400, 300, 1, False, False),
    # (2000, 0.005, 400, 300, 1, False, False),
    # (2000, 0.005, 400, 300, 1, False, False),
    # (2000, 0.005, 750, 300, 1, False, False),
    # (2000, 0.005, 750, 300, 1, False, False),
    # (2000, 0.005, 750, 300, 1, False, False),
    # (2000,0.005,150,557,2,False,False),
    # (2000,0.005,150,557,2,False,False),
    # (2000,0.005,150,557,2,False,False),
    # (2000, 0.0005, 175, 300, 2, False, False),
    # (2000, 0.005, 175, 300, 2, False, False),
    # (2000, 0.0025, 175, 300, 1, False, False),
    # (2000, 0.005, 1000, 1000, 2, False, False),
    # (2000, 0.0035, 175, 300, 2, False, False),
    # (2000, 0.005, 175, 300, 2, False, False),
    # (2000, 0.005, 175, 300, 2, False, False),
    # (2000, 0.0035, 175, 300, 2, False, False),
    # (2000, 0.005, 175, 300, 2, False, False),
    # (2000, 0.005, 350, 300, 2, False, False),
    # (2000, 0.0035, 350, 300, 1, False, False),
    # (2000, 0.005, 350, 300, 1, False, False),
    # (2000, 0.005, 500, 300, 1, False, False),
    # (2000, 0.0025, 500, 300, 1, False, False),
    # (2000, 0.005, 500, 300, 1, False, False),
    # (2000, 0.0005, 300, 300, 1, False, False),
    # (2000, 0.005, 300, 300, 1, False, False),
    # (2000, 0.0025, 300, 300, 1, False, False),
    # (2000, 0.005, 400, 300, 1, False, False),
    # (2000, 0.0035, 400, 300, 1, False, False),
    # (2000, 0.005, 400, 300, 1, False, False),
    # (2000, 0.005, 750, 300, 1, False, False),
    # (2000, 0.0035, 750, 300, 1, False, False),
    # (2000, 0.005, 750, 300, 1, False, False),
    # (2000, 0.005, 175, 300, 2, False, False),
    # (2000, 0.0035, 175, 300, 2, False, False),
    # (2000, 0.005, 175, 300, 2, False, False),
    # (2000, 0.005, 175, 300, 2, False, False),
    # (2000, 0.0035, 175, 300, 2, False, False),
    # (2000, 0.005, 175, 300, 2, False, False),
    # (2000, 0.005, 350, 300, 2, False, False),
    # (2000, 0.0035, 350, 300, 1, False, False),
    # (2000, 0.005, 350, 300, 1, False, False),
    # (2000, 0.005, 500, 300, 1, False, False),
    # (2000, 0.0025, 500, 300, 1, False, False),
    # (2000, 0.005, 500, 300, 1, False, False),
    # (2000, 0.0005, 300, 300, 1, False, False),
    # (2000, 0.005, 300, 300, 1, False, False),
    # (2000, 0.0025, 300, 300, 1, False, False),
    # (2000, 0.005, 400, 300, 1, False, False),
    # (2000, 0.0035, 400, 300, 1, False, False),
    # (2000, 0.005, 400, 300, 1, False, False),
    # (2000, 0.005, 750, 300, 1, False, False),
    # (2000, 0.0035, 750, 300, 1, False, False),
    # (2000, 0.005, 750, 300, 1, False, False),
    # ]


# path  = 'Data parsed pruned half'  
# listset = [
#     (2000,0.008,600,150,1,False,False),
#     (2000,0.0018,600,150,1,False,False),
#     (2000,0.0008,600,150,1,False,False),
#     (2000,0.004,600,150,1,False,False),
#     (2000,0.008,400,150,1,False,False),
#     (2000,0.0018,400,150,1,False,False),
#     (2000,0.0008,400,150,1,False,False),
#     (2000,0.004,400,150,1,False,False),
#     (2000,0.008,80,300,2,False,False),
#     (2000,0.0018,160,140,2,False,False),
#     (2000,0.0008,120,250,2,False,False),
#     (2000,0.004,200,200,1,False,False),
#     (2000,0.008,150,400,1,False,False),
#     (2000,0.0018,300,300,1,False,False),
#     (2000,0.0008,1000,150,1,False,False),
#     (2000,0.004,1000,200,1,False,False),
#     (2000,0.008,400,200,1,False,False),
#     (2000,0.0018,400,200,1,False,False),
#     (2000,0.0008,400,200,1,False,False),
#     (2000,0.004,400,200,1,False,False),
#     ]

# for i in range(0,25,1):
#     try:
#         epoch, decay, hidden, batch, num_layers, rescale, normalize = listset[i]
#         epoch = 500
#         lr = 0.0033
#         model = NeuralSpline(path = path,rescale_inputs = rescale)
#         model.train_val(epochs=epoch, 
#                 lr=lr, 
#                 hidden=hidden, 
#                 batch = batch,
#                 num_layers =num_layers,
#                 normalize=  normalize,decay = decay,
#                 save_path = "models/gamma_f_GRU_models_{}.pt".format(str(i)), 
#                 model_cls =GRU_model_latent_decay_simple_fb_mat)
#         model.plot_convergence(save_dir = '/home/bob-van-sluijs/Desktop/plot {} GRU half/'.format(str(i)))
#     except:
#         pass
#     try:
#         import torch, gc
#         del model  # if you still hold refs
#         gc.collect()
#         torch.cuda.empty_cache()  # frees cached blocks back to the driver
#     except:
#         pass 
    
    