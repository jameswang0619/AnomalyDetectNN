# -*- coding: utf-8 -*-
"""NNratio_1D.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13zC6nTSvjEFa_-FFy6-Rpy5zH8bUUlMJ
"""

from __future__ import print_function
from __future__ import division
import keras
import os
from keras.models import Sequential
from keras import layers
from keras import backend as K
from keras.layers import Activation, Dense
from keras.constraints import Constraint
from keras.constraints import max_norm

from math import *
import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate as integrate
import scipy.stats as st
import time

##!cat /proc/cpuinfo

##!cat /proc/meminfo

#!apt-get -qq install -y graphviz && pip install -q pydot
#import pydot

"""## Connect to Google Drive for storage

### Pydrive
"""

# install PyDrive
#!pip install -U -q PyDrive

#from pydrive.auth import GoogleAuth
#from pydrive.drive import GoogleDrive
#from google.colab import auth
#from oauth2client.client import GoogleCredentials

# 1. Authenticate and create the PyDrive client.
#auth.authenticate_user()
#gauth = GoogleAuth()
#gauth.credentials = GoogleCredentials.get_application_default()
#drive = GoogleDrive(gauth)

# PyDrive reference:
# https://googledrive.github.io/PyDrive/docs/build/html/index.html

'''
# 2. Create & upload a file 
uploaded = drive.CreateFile({'title': 'Sample upload.txt'})
uploaded.SetContentString('Sample upload file content')
uploaded.Upload()
print('Uploaded file with ID {}'.format(uploaded.get('id')))

# 3. Load a file by ID and print its contents.
downloaded = drive.CreateFile({'id': uploaded.get('id')})
print('Downloaded content "{}"'.format(downloaded.GetContentString()))
'''



"""# Main"""

#sigmoid function
def sigmoid(x):
  return 1 / (1 + exp(-x))

#@title Glabal Parameters
epochs = 100000 #@param {type:"integer"}
batch_size = 1000 #@param {type:"integer"}
#Geteps = 0.005 #@param {type:"number"}
Geteps = 0.0015811 #@param {type:"number"}
Getmu = 0.8 #@param {type:"number"}
Getsigma = 0.02 #@param {type:"number"}
cut = 0. #@param {type:"number"}
gcut = 0. #@param {type:"number"}

def SM(x):
  return exp(-8*x)

def BSM(x):
  return exp(-(x-Getmu)**2/(2*Getsigma**2))
#Normalize distribution
SM_norm = integrate.quad(lambda y :SM(y),cut,1)
BSM_norm = integrate.quad(lambda y :BSM(y),cut,1)
#SM_norm_c = integrate.quad(lambda y :SMn(y),gcut,1)

#normalized distribution
def SMn(x):
  return exp(-8*x)/SM_norm[0]

def BSMn(x):
  return (SMn(x)+Geteps*BSM(x)/BSM_norm[0])/(1+Geteps)

SM_norm_c = integrate.quad(lambda y :SM(y),gcut,1)

def SMnc(x):
  return exp(-8*x)/SM_norm_c[0]


#define probability distribution function
class P_SM(st.rv_continuous):
    def _pdf(self,x):
        return SMn(x)
      
class P_BSM(st.rv_continuous):
    def _pdf(self,x):
        return BSMn(x)        
      
class P_SMc(st.rv_continuous):
    def _pdf(self,x):
        return SMnc(x)
      
      
      
SM_gen = P_SM(a=cut,b=1,name='sm_sample')
BSM_gen = P_BSM(a=cut,b=1,name='bsm_sample')
SMc_gen = P_SMc(a=gcut,b=1,name='smc_sample')

NRef=200000
Nbsm=20032
NR =20000

#load data
samples = np.load('Sample200k_2k_R1.npz')
Ref_sample=samples['Ref_sample']
bsm_sample=samples['bsm_sample']
#Data and References
#Ref_sample = SM_gen.rvs(size=NRef)
#bsm_sample= BSM_gen.rvs(size=Nbsm)

#sm_target = np.zeros(NRef)
#bsm_target = np.ones(Nbsm)

#x_train = np.append(Ref_sample,bsm_sample)
#y_train = np.append(sm_target,bsm_target)

rfw = NRef/NR



"""## Binning the Ref sample"""
Nbins=1000
H1d,edge = np.histogram(Ref_sample,bins=Nbins,range=(0,1))
#H1d_bsm, edge_bsm = np.histogram(bsm_sample, bins=Nbins, range=(0,1))

def moving_average(a, n=2) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

xpos = moving_average(edge,2);
#xpos_bsm = moving_average(edge_bsm,2);

wlist = []
wlist_bsm = []

for xidx, xval in enumerate(xpos):
  wlist.append(H1d[xidx])

#for xidxb, xvalb in enumerate(xpos_bsm):
#  wlist_bsm.append(H1d_bsm[xidxb])


x_train = np.append(xpos,bsm_sample)
#x_train = np.append(xpos,xpos_bsm)
sm_target = np.zeros(Nbins)
bsm_target = np.ones(Nbsm)
#bsm_target = np.ones(Nbins)

y_train = np.append(sm_target,bsm_target)
#weightloss1 = np.append(np.asarray(wlist),np.asarray(wlist_bsm))
weightloss1 = np.append(np.asarray(wlist),bsm_target)
weightloss = K.variable(value=weightloss1.reshape((Nbsm+Nbins,1)))






#from pydrive.auth import GoogleAuth
#from pydrive.drive import GoogleDrive
#from google.colab import auth
#from oauth2client.client import GoogleCredentials

# 1. Authenticate and create the PyDrive client.
#auth.authenticate_user()
#gauth = GoogleAuth()
#gauth.credentials = GoogleCredentials.get_application_default()
#drive = GoogleDrive(gauth)

# PyDrive reference:
# https://googledrive.github.io/PyDrive/docs/build/html/index.html

# 2. Create & upload a file 
#uploaded = drive.CreateFile({'title': 'Samples200k_2k_R2.npz'})
#uploaded.SetContentFile('Samples200k_2k_R2.npz')
#uploaded.Upload()
#print('Uploaded file with ID {}'.format(uploaded.get('id')))

"""## Build NN and Train"""



#define custom loss function

def customloss(yTrue,yPred):
  return yTrue*K.log(1+K.exp(-yPred))+1/rfw*(1-yTrue)*K.log(1+K.exp(yPred))


def customlossML(sw):
  Nt = Nbsm+Nbins
  def lossML(yTrue,yPred):
    sw_rs = K.reshape(sw,(Nt,1))
    ytrue_rs = K.reshape(yTrue,(Nt,1))
    ypred_rs = K.reshape(yPred,(Nt,1))
    return -K.sum(ytrue_rs*sw_rs*ypred_rs)+K.sum((1-ytrue_rs)*sw_rs*(K.exp(ypred_rs)-1))/rfw
  return lossML

def customlossMLws(yTrue,yPred):
  return -yTrue*yPred+1/rfw*(1-yTrue)*(K.exp(yPred)-1)

rmsprop = keras.optimizers.RMSprop()

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

# build the model
model = Sequential()
model.add(Dense(4, activation='sigmoid',input_shape=(1,),kernel_regularizer=keras.regularizers.l2(0.),W_constraint = WeightClip(40),b_constraint=WeightClip(40)))
#model.add(Dense(4, activation='sigmoid',input_shape=(1,),kernel_regularizer=keras.regularizers.l2(0.0),kernel_constraint=max_norm(10.),bias_constraint=max_norm(10.)))
#model.add(Dense(4, activation='sigmoid',input_shape=(1,),kernel_regularizer=keras.regularizers.l2(0.0001)))
#model.add(Dense(3, activation='sigmoid'))
#model.add(Dense(3, activation='sigmoid'))
model.add(Dense(1,W_constraint=WeightClip(40)))



# compile the model
model.compile(loss=customlossML(weightloss),
              optimizer='rmsprop')

#model.save_weights("model_4_1d.h5")

'''
#Visualize network
#from keras.utils import plot_model
#plot_model(model,to_file='model.png',show_shapes='True')
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))
'''

#load weight and continue training
#import h5py

#sample1
#model.load_weights("modelML_9_cut5_D400_200_L2R0_MN0_sigmoid_2000k.h5")

#sample2
#model.load_weights("model_5_R2_R500k.h5")




"""### test statistics vs. Run"""

def t_vs_run(drun,ite):
    t_v_array=[]
    i=0;
    while i < ite:
      model.fit(x_train, y_train,
          batch_size=len(x_train),
          epochs=drun,
          shuffle=False,
          verbose=0);
      #tary = 2*(model.predict(np.column_stack((bsm_cut,anglesBSM_c)))).flatten();
      tary1 = 2*(model.predict(bsm_sample)).flatten();
      tary3 = 2*np.vectorize(exp)(model.predict(Ref_sample)).flatten();
      tary2 = -2*(model.evaluate(x=x_train,y=y_train,batch_size=len(x_train))).flatten();
      #tary3 = -2*(model.evaluate(x=bsm_sample,y=bsm_target,batch_size=len(bsm_sample))).flatten();
      t_v_array = np.append(t_v_array,np.array([np.sum(tary1),np.sum(tary2),np.sum(tary3)]));
      i +=1
    
    #model.save_weights("model1d_4.h5")
    return t_v_array

#R14
#t = time.process_time()
#ta = t_vs_run(100000,5)
#print(time.process_time() - t)

#distribution of t
#Nsmb = 2000;
#rfw_tc = len(Ref_sample)/Nsmb;

def customlossMLc(yTrue,yPred):
  return -yTrue*yPred+1/rfw_tc*(1-yTrue)*(K.exp(yPred)-1)

def customlossMLc(sw,Nt):
  def lossML(yTrue,yPred):
    sw_rs = K.reshape(sw,(Nt,1))
    ytrue_rs = K.reshape(yTrue,(Nt,1))
    ypred_rs = K.reshape(yPred,(Nt,1))
    return -K.sum(ytrue_rs*sw_rs*ypred_rs)+K.sum((1-ytrue_rs)*sw_rs*(K.exp(ypred_rs)-1))/rfw
  return lossML

class test_statistics:
  '''
  Compute test statistics given data sample(bsm or sm) 
  '''
  def __init__(self,NNmodel):
    #self.input_sample = input_sample
    #self.ref_sample = ref_sample
    self.NNmodel = NNmodel
    #self.lossfunc = lossfunc
    #self.data_sample_t = SMc_gen.rvs(size=2000)
    #self.wlist_d = []
    #self.anglesSM_t = angle_gen.rvs(size=Nsmb)
    
  #train NN with data(input) sample and ref sample
  #training variables
  
  #generate sm sample
  
  def train_t(self,epoch):
    Nsmb = np.random.poisson(Nbsm,1)
    #Nsmb = np.random.poisson(NR,1)
    #sm ref data sample
    #self.data_sample_t = SMc_gen.rvs(size=Nsmb[0])
    #bsm data sample
    #self.data_sample_t = BSM_gen.rvs(size=Nsmb[0])
    self.data_sample_t = BSM_gen.rvs(size=Nbsm)
    #bin the data sample
    self.H1d_d, self.edge_d = np.histogram(self.data_sample_t, bins=Nbins, range=(0,1))
    self.xpos_d = moving_average(self.edge_d,2);
    #bsm_target_tc = np.ones(Nsmb[0])
    bsm_target_tc = np.ones(Nbins)
    #x_train_t = np.append(xpos,self.data_sample_t)
    x_train_t = np.append(xpos,self.xpos_d)
    self.wlist_d = []
    for xidx, xval in enumerate(self.xpos_d):
      self.wlist_d.append(self.H1d_d[xidx])
    
    #bsm_target_t = np.ones(Nbsm_c)
    y_train_t = np.append(sm_target,bsm_target_tc)
    weightloss_t = np.append(np.asarray(wlist),self.wlist_d)
    #weightloss_t = np.append(np.asarray(wlist),bsm_target_tc)
    #weightlossrs = K.variable(value=weightloss_t.reshape((Nsmb[0]+Nbins,1)))
    weightlossrs = K.variable(value=weightloss_t.reshape((2*Nbins,1)))
    self.NNmodel.compile(loss=customlossMLc(weightlossrs,2*Nbins),optimizer=rmsprop)
    #self.NNmodel.load_weights("modelML_4_1d_init.h5")
    self.NNmodel.load_weights("model_4_1d.h5")
    self.NNmodel.fit(x_train_t, y_train_t,
          batch_size=len(x_train_t),
          epochs=epoch,
          shuffle=False,
          verbose=0);
    #tary = 2*(self.NNmodel.predict(self.sm_sample_t)).flatten();
    tary = -2*(model.evaluate(x=x_train_t,y=y_train_t,batch_size=len(x_train_t))).flatten();
    return (np.sum(tary))
    
  def train_result(self):
    yorig = []
    for xi in xinput:
      yorig.append(SMn(xi))
  
    ypred = np.vectorize(exp)(model.predict(xinput));
    ypred = ypred.flatten()*np.vectorize(SMn)(xinput);
    #fig, ax = plt.subplots(projection='3d')
    
    fig, ax = plt.subplots()
    ax.plot(xinput,yorig,'*')
    ax.plot(xinput,ypred,'.')
    ax.hist(self.sm_sample_t, bins=25)
    plt.yscale('log', nonposy='clip')
    plt.title("SM: 1k, BSM: 100, Sample 1, 1M Rounds")
    
    
  def t_vs_run(self,drun,ite):
    Nsmb = np.random.poisson(Nbsm,1)
    x_train_t = np.append(xpos,self.sm_sample_t)
    bsm_target_tc = np.ones(Nsmb[0])
    y_train_t = np.append(sm_target,bsm_target_tc)
    weightloss_t = np.append(wlist,bsm_target_tc)
    weightlossrs = K.variable(value=weightloss_t.reshape((Nsmb[0]+Nbins,1)))
    self.NNmodel.compile(loss=customlossMLc(weightlossrs,Nsmb[0]+Nbins),optimizer=rmsprop)
    
    t_v_array=[]
    i=0;
    while i < ite:
      self.NNmodel.fit(x_train_t, y_train_t,
          batch_size=len(x_train_t),
          epochs=drun,
          shuffle=False,
          verbose=0);
      #tary = 2*(self.NNmodel.predict(self.sm_sample_t)).flatten();
      tary = -2*(model.evaluate(x=x_train_t,y=y_train_t,batch_size=len(x_train_t))).flatten();
      t_v_array = np.append(t_v_array,np.sum(tary));
      i +=1
      
    return t_v_array
  
  def t_value(self):
    tary = 2*(self.NNmodel.predict(np.column_stack((sm_sample_t,anglesSM_t)))).flatten();
    return (np.sum(tary)) 

"""# Save and Continue training"""

#load weight and continue training
#model.load_weights("model_5_3.h5")

#Save weight to HDF5
#model.save_weights("model2.h5")
#print("Save model to disk")

#sfilename = 't_value'
#Sample200k_4k.npz
#np.savez(sfilename,t_value=ta)
'''
model.fit(x_train, y_train,
          batch_size=len(x_train),
          epochs=15000,
          sample_weight=weightloss,
          verbose=0)
      #tary = 2*(model.predict(np.column_stack((bsm_cut,anglesBSM_c)))).flatten();
tary = 2*(model.predict(bsm_sample)).flatten()
#print(np.sum(tary))
'''
#model_weight=model.get_weights()
#smtrain1 = test_statistics(model,customlossMLc)
#ta=smtrain1.t_vs_run(10000,200)

def get_tsm(Nsample):
  tvalue_array = []
  smtrain = test_statistics(model)
  i=0
  while i < Nsample:
    tvalue_array.append(smtrain.train_t(800000))
    i += 1
    
  return tvalue_array

tarrayR1 = get_tsm(5)
#f = open('tbsm20kbin_Ref1kbin_R1M_08.txt','w')
f = open('tbsm20kbin_np_fixed_p_Ref1kbin_R1M_02.txt','w')
#f.write('{}'.format(model_weight))
#f.write('{}'.format(ta))
f.write('{}'.format(tarrayR1))
f.close()

