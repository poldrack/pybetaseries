#!/usr/bin/env python

# TO DO:
# use higher temporal resolution for convolution

from mvpa.misc.fsl.base import *

import numpy as N
import nibabel
from scipy.ndimage import convolve1d
from scipy.sparse import spdiags
from scipy.linalg import toeplitz
from mvpa.datasets.mri import *
from mvpa.measures.glm import GLM
from estimate_OLS import *
import os
from copy import copy
from spm_hrf import spm_hrf

#def estimate_lsone(fsfdir,method='lsone',time_res=0.1):
if 1==1: 
    fsfdir='/Users/poldrack/data/python-ls/testdata3.feat/'
    methods={}
    methods['lsall']=0
    methods['lsone']=1
    #TR=2
    if not os.path.exists(fsfdir):
        print 'ERROR: %s does not exist!'%fsfdir
        #return
    
    if fsfdir[-1]!='/':
        fsfdir=fsfdir+'/'


    # load design

    fsffile=fsfdir+'design.fsf'
    desmatfile=fsfdir+'design.mat'

    design=read_fsl_design(fsffile)

    desmat=FslGLMDesign(desmatfile)
    nevs=desmat.mat.shape[1]
    ntp=desmat.mat.shape[0]

    TR=design['fmri(tr)']
    if TR != 2.0:
        print 'ERROR: currently only works with TR=2.0'
        #return

    time_res=0.1  # time resolution for upsampled design matrix
    hrf=spm_hrf(time_res)
    collapse_other_conditions=1
    
    time_up=N.arange(0,TR*ntp+time_res, time_res);
    max_evtime=TR*ntp - 2;
    
    n_up=len(time_up)
    good_evs=[]
    motion_evs=[]
    ons=[]
    
    if not os.path.exists(fsfdir+'betaseries'):
        os.mkdir(fsfdir+'betaseries')

    # load data
    #maskimg='/Users/poldrack/data/python-ls/mask_vox_20_35_16.nii.gz'
    maskimg=fsfdir+'mask.nii.gz'
    data=fmri_dataset(fsfdir+'filtered_func_data.nii.gz',mask=maskimg)
    mask=fmri_dataset(maskimg,mask=maskimg)
    nvox=data.nfeatures


    # create smoothing kernel for design

    cutoff=design['fmri(paradigm_hp)']/TR
    sigN2=(cutoff/(N.sqrt(2.0)))**2.0;
    K=toeplitz(1/N.sqrt(2.0*N.pi*sigN2)*N.exp((-1*N.array(range(ntp))**2.0/(2*sigN2))))
    K=spdiags(1./N.sum(K.T,0).T, 0, ntp,ntp)*K;
    H = N.zeros((ntp,ntp)) # Smoothing matrix, s.t. H*y is smooth line 
    X = N.hstack((N.ones((ntp,1)), N.arange(1,ntp+1).T[:,N.newaxis]));
    for  k in range(ntp):
           W = N.diag(K[k,:])
           Hat = N.dot(N.dot(X,N.linalg.pinv(N.dot(W,X))),W)
           H[k,:] = Hat[k,:]

    F=N.eye(ntp)-H

    # loop through and find the good (non-motion) conditions

    evctr=0
    ev_td=N.zeros(design['fmri(evs_real)'])

    for ev in range(design['fmri(evs_orig)']):
        # filter out motion parameters
        if design['fmri(evtitle%d)'%int(ev+1)].find('motpar')<0:
            good_evs.append(evctr)
            evctr+=1
            if design['fmri(deriv_yn%d)'%int(ev+1)]==1:
                ev_td[evctr]=1
                # skip temporal derivative
                evctr+=1
            ons.append(FslEV3(fsfdir+'custom_timing_files/ev%d.txt'%int(ev+1)))
        else:
            motion_evs.append(evctr)
            evctr+=1
            if design['fmri(deriv_yn%d)'%int(ev+1)]==1:
                # skip temporal derivative
                ev_td[evctr]=1
                motion_evs.append(evctr)
                evctr+=1
    ntrials_total=0
    for x in range(len(good_evs)):
        ntrials_total=ntrials_total+len(ons[x]['onsets'])
    
    # now loop through the good evs and build the ls-one model
    # design matrix for each trial/ev
    if methods['lsone']==1:
       method='lsone'
       print 'estimating ls-one model...'
       dm_nuisanceevs=desmat.mat[:,motion_evs]
       trial_ctr=0
       all_conds=[]
       beta_maker=N.zeros((ntrials_total,ntp))
       for e in range(len(good_evs)):
            ev=good_evs[e]
            # first, take the original desmtx and remove the ev
            other_good_evs=[x for x in good_evs if x != ev]
            # put the temporal derivatives in
            og=copy(other_good_evs)
            for x in og:
                if ev_td[x]>0:
                    other_good_evs.append(x+1)
            dm_otherevs=desmat.mat[:,other_good_evs]
            cond_ons=N.array(ons[e].onsets)
            cond_dur=N.array(ons[e].durations)
            # cond_ons=N.round(cond_ons/TR)  # round to nearest TR number
            ntrials=len(cond_ons)
            glm_res_full=N.zeros((nvox,ntrials))
            print 'processing ev %d: %d trials'%(e+1,ntrials)
            #print cond_ons
            for t in range(ntrials):
                all_conds.append((ev/2)+1)
                #print 'processing ev %d trial %d onset %f'%(ev,t,cond_ons[t])
                #print 'processing TOI'
                if cond_ons[t] > max_evtime:
                    print 'TOI: skipping ev %d trial %d: %f %f'%(ev,t,cond_ons[t],max_evtime)
                    trial_ctr+=1
                    continue
                # first build model for the trial of interest at high resolution
                dm_toi=N.zeros(n_up)
                window_ons=[N.where(time_up==x)[0][0] for x in time_up if (x > cond_ons[t]) & (x < cond_ons[t] + cond_dur[t])]
                dm_toi[window_ons]=1
                dm_toi=N.convolve(dm_toi,hrf)[0:ntp/time_res*TR:(TR/time_res)]
                other_trial_ons=cond_ons[N.where(cond_ons!=cond_ons[t])[0]]
                other_trial_dur=cond_dur[N.where(cond_ons!=cond_ons[t])[0]]
                
                dm_other=N.zeros(n_up)
                #print 'processing other trials'
                for o in other_trial_ons:
                    if o > max_evtime:
                        #print 'OTHER: skipping ev %d trial %d: %f %f'%(ev,t,o,max_evtime)
                        continue
                    window_ons=[N.where(time_up==x)[0][0] for x in time_up if (x > o) & (x < o + other_trial_dur[N.where(other_trial_ons==o)[0][0]])]
                    dm_other[window_ons]=1
                #print 'making design matrix'
                dm_other=N.convolve(dm_other,hrf)[0:ntp/time_res*TR:(TR/time_res)]
                if collapse_other_conditions:
                    dm_other=N.hstack((N.dot(F,dm_other[0:ntp,N.newaxis]),dm_otherevs))
                    dm_other=N.sum(dm_other,1)
                    dm_full=N.hstack((N.dot(F,dm_toi[0:ntp,N.newaxis]),dm_other[:,N.newaxis],dm_nuisanceevs))
                else:
                    dm_full=N.hstack((N.dot(F,dm_toi[0:ntp,N.newaxis]),N.dot(F,dm_other[0:ntp,N.newaxis]),dm_otherevs,dm_nuisanceevs))
                dm_full=dm_full - N.kron(N.ones((dm_full.shape[0],dm_full.shape[1])),N.mean(dm_full,0))[0:dm_full.shape[0],0:dm_full.shape[1]]
                dm_full=N.hstack((dm_full,N.ones((ntp,1))))
                beta_maker_loop=N.linalg.pinv(dm_full)
                beta_maker[trial_ctr,:]=beta_maker_loop[0,:]
                trial_ctr+=1
                
       glm_res_full=N.dot(beta_maker,data.samples)
       #glm_res_full=glm_res_full[0:ntrials,:]

     #           for v in range(nvox):
                    #try:
     #               glm_result=estimate_OLS(dm_full,data.samples[:,v])
     #               glm_res_full[v,t]=glm_result[0]
                    #except:
                    #    print 'problem with trial %d, cond %d'%(t,e)
       all_conds=N.array(all_conds)
       for e in range(len(good_evs)):
           ni=map2nifti(data,data=glm_res_full[N.where(all_conds==(e+1))[0],:])
           ni.to_filename(fsfdir+'betaseries/ev%d_%s.nii.gz'%(int(e+1),method))


    if methods['lsall']==1:  # do ls-all
       method='lsall'
       print 'estimating ls-all...'
       dm_nuisance=desmat.mat[:,motion_evs]
       # first get all onsets in a row
       all_onsets=[]
       all_durations=[]
       all_conds=[]  # condition marker
       for e in range(len(good_evs)):
            ev=good_evs[e]        
            all_onsets=N.hstack((all_onsets,ons[e].onsets))
            all_durations=N.hstack((all_durations,ons[e].durations))
            all_conds=N.hstack((all_conds,N.ones(len(ons[e].onsets))*((ev/2)+1)))

       #all_onsets=N.round(all_onsets/TR)  # round to nearest TR number
       ntrials=len(all_onsets)
       glm_res_full=N.zeros((nvox,ntrials))
       dm_trials=N.zeros((ntp,ntrials))
       dm_full=[]
       for t in range(ntrials):
                if all_onsets[t] > max_evtime:
                    continue
                # build model for each trial
                dm_trial=N.zeros(n_up)
                window_ons=[N.where(time_up==x)[0][0] for x in time_up if (x > all_onsets[t]) & (x < all_onsets[t] + all_durations[t])]
                dm_trial[window_ons]=1
                dm_trial=N.convolve(dm_trial,hrf)[0:ntp/time_res*TR:(TR/time_res)]
                dm_trials[:,t]=dm_trial

       # filter the desmtx, except for the nuisance part (which is already filtered)
       if len(motion_evs)>0:
           dm_full=N.hstack((N.dot(F,dm_trials),dm_nuisance))
       else:
           dm_full=N.dot(F,dm_trials)

       dm_full=dm_full - N.kron(N.ones((dm_full.shape[0],dm_full.shape[1])),N.mean(dm_full,0))[0:dm_full.shape[0],0:dm_full.shape[1]]
       dm_full=N.hstack((dm_full,N.ones((ntp,1))))
       glm_res_full=N.dot(N.linalg.pinv(dm_full),data.samples)
       glm_res_full=glm_res_full[0:ntrials,:]

       #for v in range(nvox):
                    #try:
        #                glm_result=estimate_OLS(dm_full,data.samples[:,v])
         #               glm_res_full[v,:]=glm_result[0:ntrials]
                    #except:
                    #    print 'problem with trial %d, cond %d'%(t,e)


       for e in range(len(good_evs)):
           ni=map2nifti(data,data=glm_res_full[N.where(all_conds==(e+1))[0],:])
           ni.to_filename(fsfdir+'betaseries/ev%d_%s.nii.gz'%(int(e+1),method))

