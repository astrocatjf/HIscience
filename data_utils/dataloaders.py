from __future__ import absolute_import, division, print_function
import numpy as np
from tensorflow import keras
from tensorflow.python.client import device_lib
from tensorflow.python.keras.utils.data_utils import Sequence
import h5py
import os
import healpy as hp

def rotate_map(hmap, rot_theta, rot_phi):
    """
    Take hmap (a healpix map array) and return another healpix map array 
    which is ordered such that it has been rotated in (theta, phi) by the 
    amounts given.
    """
    nside = hp.npix2nside(len(hmap))

    # Get theta, phi for non-rotated map
    t,p = hp.pix2ang(nside, np.arange(hp.nside2npix(nside))) #theta, phi

    # Define a rotator
    r = hp.Rotator(deg=False, rot=[rot_phi,rot_theta])

    # Get theta, phi under rotated co-ordinates
    trot, prot = r(t,p)

    # Interpolate map onto these co-ordinates
    rot_map = hp.get_interp_val(hmap, trot, prot)

    return rot_map

class dataLoaderDeep21(Sequence):
    'Generates random subset of simulation data per epoch for Keras'
    def __init__(self, 
                 path, 
                 data_type='train', 
                 is_3d = True,
                 batch_size=48, 
                 num_sets=1,
                 sample_size=20,
                 shuffle=False,
                 bin_min = 1,
                 bin_max = 160, 
                 nu_indx=None,
                 nu_skip=1,
                 aug = True,
                 stoch = True,
                 nwinds = 768
                ):
        
        
        'Initialization'
        self.data_type = data_type
        self.is_3d = is_3d
        self.sample_size = sample_size
        self.num_sets = num_sets
        self.nwinds = nwinds          # simulation param, num bricks per sim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bin_min = bin_min - 1 # python indexes from 0
        self.bin_max = bin_max - 1
        self.path = path
        self.stoch = stoch
        self.aug = aug
        self.fname = path +  'dataset_%d.h5'%(int(np.ceil(np.random.rand()*self.num_sets)))
        self.datafile = h5py.File(self.fname, 'r')[self.data_type]
        self._dat_length = len(self.datafile)
        self.indexes = np.arange(self.nwinds*self.sample_size)
        self.nu_indx = nu_indx
        self.nu_skip = nu_skip
        self.on_epoch_end()


    def __len__(self):
        return int(np.floor(((self.sample_size)*self.nwinds) // self.batch_size))

    def __getitem__(self, idx):
        #fname = self.path +  'dataset_%d.h5'%(int(np.ceil(np.random.rand()*self.num_sets)))
        #ind = self.indexes[idx]  # index in terms of shuffled data
        x,y = self.load_data(idx)
        
        return x,y
    
    def on_epoch_end(self):
        # switch up dataset every other time to change noise
        if np.random.rand() > 0.5:
            #self.datafile.close()
            self.fname = self.path +  'dataset_%d.h5'%(int(np.ceil(np.random.rand()*self.num_sets)))
            self.datafile = h5py.File(self.fname, 'r')[self.data_type]

        # draw randomly with replacement but keep dataset fixed
        if self.stoch:

            inds = np.arange(self._dat_length).reshape(-1, self.nwinds)
            l = np.arange(len(inds))
            # choose random skies with replacement
            inds = inds[np.random.choice(l, size=self.sample_size)]
            # rotate each selected map
            #if self.aug:
                # choose some random angles between +/- 2pi to rotate by, one for each in sample_size
               # r = np.random.uniform(low=-2, high=2, size=self.sample_size)
               # inds = np.concatenate([rotate_map(inds[i].flatten(), r[i]*np.pi, 0) for i in range(self.sample_size)])
             #   pass
                # reshape array
            self.indexes = inds.reshape(self.sample_size*self.nwinds)

    
    
    def load_data(self, idx):
        #self.datafile = h5py.File(self.fname, 'r')
        # print('self.batch_size: ', self.batch_size)
        # print('self.indexes: ', self.indexes)
        # print('idx: ', idx)
        d = self.datafile[self.indexes[idx*self.batch_size:(idx+1)*self.batch_size],:,:,:,:]
        x = d.T[0].T
        y = d.T[1].T; del d
        #self.datafile.close()
        

        # rearrange frequencies if desired with input indexes
        if self.nu_indx is not None:
            x = x.T[self.nu_indx].T       
            y = y.T[self.nu_indx].T   
        
        if self.is_3d:
            x = np.expand_dims(x, axis=-1)
            y = np.expand_dims(y, axis=-1)

        return x,y
