#!/usr/bin/env python

# implementation of spm_hrf.m from SPM distribution
import scipy.stats
import numpy as N

def spm_hrf(TR,p=[6,16,1,1,6,0,32]):
## % RT   - scan repeat time
## % p    - parameters of the response function (two gamma functions)
## %
## %                                                     defaults
## %                                                    (seconds)
## %   p(1) - delay of response (relative to onset)         6
## %   p(2) - delay of undershoot (relative to onset)      16
## %   p(3) - dispersion of response                        1
## %   p(4) - dispersion of undershoot                      1
## %   p(5) - ratio of response to undershoot               6
## %   p(6) - onset (seconds)                               0
## %   p(7) - length of kernel (seconds)                   32

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
