from __future__ import print_function

from __future__ import division

import keras

import os

import time

from keras.models import Sequential, Model

from keras import layers

from keras import backend as K

from keras import optimizers

from keras.layers import Activation, Dense, Merge

from keras.constraints import Constraint

from math import *
import numpy as np
%matplotlib inline
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate as integrate
import scipy.stats as st
from keras.constraints import max_norm






class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''
    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}


class test_statistics:
    '''
    Compute test statistics given data sample(bsm or sm)
    '''

    def __init__(self, nnmodel, lossfunc, refsample):
        self.nnmodel = nnmodel
        self.lossfunc = lossfunc
        self.refsample = refsample
        self.sm_sample_t = SMc_gen.rvs(size=Nsmb)
        self.anglesSM_t = angle_gen.rvs(size=Nsmb)

    # train NN with data(input) sample and ref sample
    # training variables

    # generate sm sample

    def train_t(self, epoch):
        self.sm_sample_t = SMc_gen.rvs(size=Nsmb)
        self.anglesSM_t = angle_gen.rvs(size=Nsmb)
        bsm_target_tc = np.ones(Nsmb)
        xmass_t = np.append(sm_cut, self.sm_sample_t)
        xangle_t = np.append(anglesSM_c, self.anglesSM_t)

        # bsm_target_t = np.ones(Nbsm_c)

        x_train_t = np.column_stack((xmass_t, xangle_t))
        y_train_t = np.append(sm_target_c, bsm_target_tc)
        self.NNmodel.compile(loss=lossfunc, optimizer=rmsprop)
        # self.NNmodel.load_weights("modelML_5_2d_init.h5")
        self.NNmodel.fit(x_train_t, y_train_t,
                         batch_size=len(x_train_t),
                         epochs=epoch,
                         verbose=0);
        tary = 2 * (self.NNmodel.predict(np.column_stack((self.sm_sample_t, self.anglesSM_t)))).flatten();
        return (np.sum(tary))

    def train_result(self, cutv):
        x = np.arange(cutv, 1, 0.01)
        y = np.arange(-pi / 2, pi / 2, 0.025)

        tseq = []
        for j in y:
            for i in x:
                tseq = np.append(tseq, exp(model.predict(np.asarray([[i, j]]))) * SM2d_c(i, j, cutv))

        tseqr = np.reshape(tseq, (len(y), len(x)))
        # fig, ax = plt.subplots(projection='3d')

        fig1 = plt.figure(figsize=(7, 7))
        xx, yy = np.meshgrid(x, y, sparse=True)
        ax = fig1.add_subplot(111, projection='3d')
        ax.plot_surface(xx, yy, tseqr)
        ax.set_xlabel('x')
        ax.set_ylabel(r'$\theta$')
        plt.title("SM: 2k, BSM: 200, Sample 1, 1.2M Rounds", fontsize=14)
        plt.show()

    def t_vs_run(self, drun, ite):
        xmass_t = np.append(sm_cut, self.sm_sample_t)
        xangle_t = np.append(anglesSM_c, self.anglesSM_t)
        bsm_target_tc = np.ones(Nsmb)
        x_train_t = np.column_stack((xmass_t, xangle_t))
        y_train_t = np.append(sm_target_c, bsm_target_tc)
        self.NNmodel.compile(loss=self.lossfunc, optimizer=rmsprop)

        t_v_array = []
        i = 0;
        while i < ite:
            self.NNmodel.fit(x_train_t, y_train_t,
                             batch_size=len(x_train_t),
                             epochs=drun,
                             verbose=0);
            tary = 2 * (self.NNmodel.predict(np.column_stack((self.sm_sample_t, self.anglesSM_t)))).flatten();
            t_v_array = np.append(t_v_array, np.sum(tary));
            i += 1

        return t_v_array

    def t_value(self):
        tary = 2 * (self.NNmodel.predict(np.column_stack((sm_sample_t, anglesSM_t)))).flatten();
        return (np.sum(tary))


    def main(self):






    if __name__ =='__main__':
        main()