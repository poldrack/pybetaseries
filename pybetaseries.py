#!/usr/bin/env python
"""pybetaseries: a module for computing beta-series regression on fMRI data

Includes:
pybetaseries: main function
spm_hrf: helper function to generate double-gamma HRF
"""

#from mvpa.misc.fsl.base import *
from mvpa2.misc.fsl.base import *
from mvpa2.datasets.mri import fmri_dataset

import numpy as N
import nibabel
import scipy.stats
from scipy.ndimage import convolve1d
from scipy.sparse import spdiags
from scipy.linalg import toeplitz
from mvpa2.datasets.mri import *
import os, sys
from copy import copy
import argparse

class pybetaseries():
    
    def __init__(self,fsfdir,tempderiv,motpars,time_res):
        #Sets up dataset wide variables
        
        self.tempderiv=tempderiv
        self.motpars=motpars
        self.time_res=time_res
        
        if not os.path.exists(fsfdir):
            print 'ERROR: %s does not exist!'%fsfdir
            
        if not fsfdir.endswith('/'):
            fsfdir=''.join([fsfdir,'/'])
        
        self.fsfdir=fsfdir
    
        fsffile=''.join([self.fsfdir,'design.fsf'])
        desmatfile=''.join([self.fsfdir,'design.mat'])
    
        design=read_fsl_design(fsffile)
    
        self.desmat=FslGLMDesign(desmatfile)
        
        self.nevs=self.desmat.mat.shape[1]
        self.ntp=self.desmat.mat.shape[0]
        
        self.TR=round(design['fmri(tr)'],2)
    
        self.hrf=spm_hrf(self.time_res)

        self.time_up=N.arange(0,self.TR*self.ntp+self.time_res, self.time_res);
        
        self.max_evtime=self.TR*self.ntp - 2;
        self.n_up=len(self.time_up)
        
        
        
        if not os.path.exists(fsfdir+'betaseries'):
            os.mkdir(fsfdir+'betaseries')
    
        # load data
        
        maskimg=''.join([fsfdir,'mask.nii.gz'])
        self.raw_data=fmri_dataset(fsfdir+'filtered_func_data.nii.gz',mask=maskimg)
        voxmeans = N.mean(self.raw_data.samples,axis=0)
        self.data=self.raw_data.samples-voxmeans
        
        self.nvox=self.raw_data.nfeatures
        
        cutoff=design['fmri(paradigm_hp)']/self.TR
        
        
        self.F=get_smoothing_kernel(cutoff, self.ntp)
            
    def LSS(self,whichevs,numrealev):
        method='LSS'
        print "Calculating LSS: Ev %s" %(whichevs[0])
        
        nuisance = otherevs(whichevs,numrealev,self.tempderiv,self.motpars)    
            
        ons=FslEV3(self.fsfdir+'custom_timing_files/ev%d.txt'%int(whichevs[0]))
        
        dm_nuisanceevs = self.desmat.mat[:, nuisance]
            
        ntrials=len(ons.onsets)
        beta_maker=N.zeros((ntrials,self.ntp))
        dm_trials=N.zeros((self.ntp,ntrials))
        
        for t in range(ntrials):
            if ons.onsets[t] > self.max_evtime:
                continue
            # build model for each trial
            dm_trial=N.zeros(self.n_up)
            window_ons = [N.where(self.time_up==x)[0][0]
                      for x in self.time_up
                      if ons.onsets[t] <= x < ons.onsets[t] + ons.durations[t]]
            
            dm_trial[window_ons]=1
            dm_trial=N.convolve(dm_trial,self.hrf)[0:int(self.ntp/self.time_res*self.TR):int(self.TR/self.time_res)]
            dm_trials[:,t]=dm_trial
        
        dm_full=N.dot(self.F,dm_trials)
        dm_full=dm_full - N.kron(N.ones((dm_full.shape[0],dm_full.shape[1])),N.mean(dm_full,0))[0:dm_full.shape[0],0:dm_full.shape[1]]
        
        
            
        for p in range(len(dm_full[1,:])):
            target=dm_full[:,p]
            dmsums=N.sum(dm_full,1)-dm_full[:,p]
            des_loop=N.hstack((target[:,N.newaxis],dmsums[:,N.newaxis],dm_nuisanceevs))
            beta_maker_loop=N.linalg.pinv(des_loop)
            beta_maker[p,:]=beta_maker_loop[0,:]
                
        # this uses Jeanette's trick of extracting the beta-forming vector for each
        # trial and putting them together, which allows estimation for all trials
        # at once
        
        glm_res_full=N.dot(beta_maker,self.data)
        ni=map2nifti(self.raw_data,data=glm_res_full)
        ni.to_filename(self.fsfdir+'betaseries/ev%d_%s.nii.gz'%(int(whichevs[0]),method))
            
    def LSA(self,whichevs,numrealev):
        method='LSA'
        print "Calculating LSA"
        
        nuisance = otherevs(whichevs,numrealev,self.tempderiv,self.motpars)
        dm_nuisanceevs = self.desmat.mat[:, nuisance]
        all_onsets=[]
        all_durations=[]
        all_conds=[]  # condition marker
        for e in range(len(whichevs)):
            ev=whichevs[e]
            ons=FslEV3(self.fsfdir+'custom_timing_files/ev%d.txt'%int(ev))       
            all_onsets=N.hstack((all_onsets,ons.onsets))
            all_durations=N.hstack((all_durations,ons.durations))
            all_conds=N.hstack((all_conds,N.ones(len(ons.onsets))*(ev)))
        ntrials=len(all_onsets)
        glm_res_full=N.zeros((self.nvox,ntrials))
        dm_trials=N.zeros((self.ntp,ntrials))
        dm_full=[]
        for t in range(ntrials):
            if all_onsets[t] > self.max_evtime:
                continue
            dm_trial=N.zeros(self.n_up)
            window_ons = [N.where(self.time_up==x)[0][0]
                      for x in self.time_up
                      if all_onsets[t] <= x < all_onsets[t] + all_durations[t]]
            
            dm_trial[window_ons]=1
            dm_trial=N.convolve(dm_trial,self.hrf)[0:int(self.ntp/self.time_res*self.TR):int(self.TR/self.time_res)]
            dm_trials[:,t]=dm_trial
            
        dm_full=N.dot(self.F,dm_trials)
        dm_full=dm_full - N.kron(N.ones((dm_full.shape[0],dm_full.shape[1])),N.mean(dm_full,0))[0:dm_full.shape[0],0:dm_full.shape[1]]
        
        
        dm_full=N.hstack((dm_full,dm_nuisanceevs))
        glm_res_full=N.dot(N.linalg.pinv(dm_full),self.data)
        glm_res_full=glm_res_full[0:ntrials,:]
    
        for e in whichevs:
            ni=map2nifti(self.raw_data,data=glm_res_full[N.where(all_conds==(e))[0],:])
            ni.to_filename(self.fsfdir+'betaseries/ev%d_%s.nii.gz'%(int(e),method))

        
def get_smoothing_kernel(cutoff, ntp):
    sigN2 = (cutoff/(N.sqrt(2.0)))**2.0
    K = toeplitz(1
                 /N.sqrt(2.0*N.pi*sigN2)
                 *N.exp((-1*N.array(range(ntp))**2.0/(2*sigN2))))
    K = spdiags(1./N.sum(K.T, 0).T, 0, ntp, ntp)*K
    H = N.zeros((ntp, ntp)) # Smoothing matrix, s.t. H*y is smooth line
    X = N.hstack((N.ones((ntp, 1)), N.arange(1, ntp+1).T[:, N.newaxis]))
    for  k in range(ntp):
        W = N.diag(K[k, :])
        Hat = N.dot(N.dot(X, N.linalg.pinv(N.dot(W, X))), W)
        H[k, :] = Hat[k, :]

    F = N.eye(ntp) - H
    return F

def otherevs(whichevs,numrealev,tempderiv,motpars):
        #sets up the onsets and nuisance EVs for given target EV 
        
        if tempderiv:
            nuisance=range(0,2*numrealev)
            popevs=[(ev-1)*2 for ev in whichevs]
            nuisance=[i for i in nuisance if i not in popevs]

        
            if motpars:
                nuisance.extend(range(2*numrealev,(6*2+2*numrealev)))
        
        
        else:
            nuisance=range(0,numrealev)
            popevs=[(ev-1) for ev in whichevs]
            nuisance=[i for i in nuisance if i not in popevs]
        
            if motpars:
                nuisance.extend(range(numrealev,6+numrealev))
        
        return nuisance



def spm_hrf(TR,p=[6,16,1,1,6,0,32]):
    """ An implementation of spm_hrf.m from the SPM distribution

    Arguments:

    Required:
    TR: repetition time at which to generate the HRF (in seconds)

    Optional:
    p: list with parameters of the two gamma functions:
                                                         defaults
                                                        (seconds)
       p[0] - delay of response (relative to onset)         6
       p[1] - delay of undershoot (relative to onset)      16
       p[2] - dispersion of response                        1
       p[3] - dispersion of undershoot                      1
       p[4] - ratio of response to undershoot               6
       p[5] - onset (seconds)                               0
       p[6] - length of kernel (seconds)                   32

    """

    p=[float(x) for x in p]

    fMRI_T = 16.0

    TR=float(TR)
    dt  = TR/fMRI_T
    u   = N.arange(p[6]/dt + 1) - p[5]/dt
    hrf=scipy.stats.gamma.pdf(u,p[0]/p[2],scale=1.0/(dt/p[2])) - scipy.stats.gamma.pdf(u,p[1]/p[3],scale=1.0/(dt/p[3]))/p[4]
    good_pts=N.array(range(N.int(p[6]/TR)))*fMRI_T
    hrf=hrf[list(good_pts)]
    # hrf = hrf([0:(p(7)/RT)]*fMRI_T + 1);
    hrf = hrf/N.sum(hrf);
    return hrf


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--fsldir',dest='fsldir',default='',help='Path to Target FSL Directory')
    parser.add_argument('--whichevs', dest='whichevs',type=int, nargs='*',help='List of EVs to Compute Beta Series for. Number corresponds to real EV number in FEAT.')
    parser.add_argument('--numrealev',dest='numrealev',type=int,help='Total number of real EVs in Feat model')
    parser.add_argument('-motpars',dest='motpars',action='store_true',
            default=False,help='Include tag if motion parameters are to be included in model. The code assumes that motion parameters are the first 6 EVs (12 if including temporal derivative EVs) after the real EVs in the Feat design matrix.')
    parser.add_argument('-tempderiv',dest='tempderiv',action='store_true',
            default=False,help='Include tag if the original design matrix includes temporal derivates. The code assumes that temporal derivatives are immediately after each EV/motion parameter in the Feat design matrix.')
    parser.add_argument('--timeres',dest='timeres',type=float,default=0.001, help='Time resolution for convolution.')
    parser.add_argument('-LSA',dest='LSA',action='store_true',
            default=False,help='Include tag to compute LSA.')
    parser.add_argument('-LSS',dest='LSS',action='store_true',
            default=False,help='Include tag to compute LSS.')
    
    
    args = parser.parse_args()
        
    
    
        
    pybeta=pybetaseries(args.fsldir,args.tempderiv,args.motpars,args.timeres)
    if args.LSS:
        for ev in args.whichevs:
            pybeta.LSS([ev],args.numrealev)
    if args.LSA:
        pybeta.LSA(args.whichevs,args.numrealev)
        