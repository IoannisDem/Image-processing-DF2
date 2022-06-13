#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from cued_sf2_lab.dct import dct_ii
from cued_sf2_lab.lbt import pot_ii
from cued_sf2_lab.laplacian_pyramid import rowint, rowint2
from cued_sf2_lab.dwt import dwt
from cued_sf2_lab.dwt import idwt
from cued_sf2_lab.familiarisation import load_mat_img
from cued_sf2_lab.dct import colxfm
from cued_sf2_lab.familiarisation import plot_image
from cued_sf2_lab.dct import regroup
from cued_sf2_lab.laplacian_pyramid import bpp
from cued_sf2_lab.laplacian_pyramid import quantise
from scipy import optimize

X, cmaps_dict = load_mat_img(img='lighthouse.mat', img_info='X', cmap_info={'map', 'map2'})
X = X - 128.0


def split(array, nrows, ncols):
    """Split a matrix into sub-matrices."""

    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))

def img_size(img):
    return bpp(img) * img.shape[0] * img.shape[1]

def dct(X, N, step, rise2step):
    
    C = dct_ii(N)
    Y = colxfm(colxfm(X, C).T, C).T
    Yq = quantise(Y, step)
    
    Yr = regroup(Yq, N)/N
    s = Yr.shape[0]//N
    #s = Yr.shape[0]//16 # LBT only
    
    imgs = split(Yr, s, s)
    size = sum(img_size(img) for img in imgs)

def dct_opt_rise(X, s_opt, N):
    """Optimisation function. Returns both the optimal step size and its error"""
    
    # Reference Scheme
    Xq = quantise(X,17)
    ref_size = img_size(Xq)
    goal = np.std(X - Xq)
    
    # N x N block {4x4, 8x8, 16x16, ...}
    C = dct_ii(N) 

    # Apply DCT as usual
    Y = colxfm(colxfm(X, C).T, C).T
   
    def error_difference(rise):
        
        # Quantise layer
        Yq = quantise(Y, step = s_opt, rise1 = rise)
        
        # Decoded image
        Zq = colxfm(colxfm(Yq.T, C.T).T, C.T)
        
        current = np.std(X - Zq)
        error = np.abs(goal-current)

        return error

    res = optimize.minimize_scalar(error_difference, bounds = (0, 2))
    
    rise_opt = res.x
    
    # Optimised quantised image Y
    Yq_opt = quantise(Y, s_opt, rise_opt)
    
    Yr = regroup(Yq_opt, N)/N
    s = Yr.shape[0]//N
    #s = Yr.shape[0]//16 # LBT only
    
    imgs = split(Yr, s, s)
    size = sum(img_size(img) for img in imgs)
    
    CR = round(ref_size/size, 4)
    
    # Optimal decoded image
    Zq_opt = colxfm(colxfm(Yq_opt.T, C.T).T, C.T)

    return [rise_opt, CR, size]

def dct_opt_step(X, N):
    """Optimisation function. Returns both the optimal step size and its error"""
    
    # Reference Scheme
    Xq = quantise(X,17)
    ref_size = img_size(Xq)
    goal = np.std(X - Xq)
    
    # N x N block {4x4, 8x8, 16x16, ...}
    C = dct_ii(N) 

    # Apply DCT as usual
    Y = colxfm(colxfm(X, C).T, C).T
   
    def error_difference(steps):
        
        # Quantise layer
        Yq = quantise(Y, steps)
        
        # Decoded image
        Zq = colxfm(colxfm(Yq.T, C.T).T, C.T)
        
        current = np.std(X - Zq)
        error = np.abs(goal-current)

        return error

    res = optimize.minimize_scalar(error_difference, bounds=(0, 256))
    
    step_opt = res.x
    error_opt = res.fun
    
    # Optimised quantised image Y
    Yq_opt = quantise(Y, step_opt)
    
    Yr = regroup(Yq_opt, N)/N
    s = Yr.shape[0]//N
    #s = Yr.shape[0]//16 # LBT only
    
    imgs = split(Yr, s, s)
    size = sum(img_size(img) for img in imgs)
    
    CR = round(ref_size/size, 4)
    
    # Optimal decoded image
    Zq_opt = colxfm(colxfm(Yq_opt.T, C.T).T, C.T)

    
    return [step_opt, CR, Zq_opt]

def lbt_opt_step(X, N, sf = 1.34):
    """Optimisation function. Returns both the optimal step size and its error"""
    
    # Reference Scheme
    Xq = quantise(X, 17)
    ref_size = img_size(Xq)
    goal = np.std(X - Xq)
    Xp = X.copy()
    
    # N x N block {4x4, 8x8, 16x16, ...}
    C = dct_ii(N)
    
    #Slicer
    t = np.s_[N//2:-N//2] # slice(start, end, step)
    
    # Filters
    Pf, Pr = pot_ii(N, sf)
    
    # Pre-filter X before applying DCT
    Xp[t,:] = colxfm(Xp[t,:], Pf)
    Xp[:,t] = colxfm(Xp[:,t].T, Pf).T

    # Apply DCT as usual
    Yp = colxfm(colxfm(Xp, C).T, C).T
   
    def error_difference(steps):
        
        # Quantise layer
        Ypq = quantise(Yp, steps)
        
        # Decoded image
        # Apply inverse DCT
        Zpq = colxfm(colxfm(Ypq.T, C.T).T, C.T)

        # Post filter Z with Pr
        Zpq[:,t] = colxfm(Zpq[:,t].T, Pr.T).T
        Zpq[t,:] = colxfm(Zpq[t,:], Pr.T)
        
        current = np.std(X - Zpq)
        error = np.abs(goal-current)

        return error

    res = optimize.minimize_scalar(error_difference, bounds=(0, 256))
    
    step_opt = res.x
    error_opt = res.fun
    
    Yq_opt = quantise(Yp, step_opt)
    
    Yr = regroup(Yq_opt, N)/N
    #s = Yr.shape[0]//N
    s = Yr.shape[0]//16 # only split with 16x16 blocks
    
    imgs = split(Yr, s, s)
    size = sum(img_size(img) for img in imgs)
    CR = round(ref_size/size, 4)    
    
    #Reconstruct optimised image
    # Apply inverse DCT
    Zpq = colxfm(colxfm(Yq_opt.T, C.T).T, C.T)

    # Post filter Z with Pr
    Zpq[:,t] = colxfm(Zpq[:,t].T, Pr.T).T
    Zpq[t,:] = colxfm(Zpq[t,:], Pr.T)
    
    return [step_opt, CR, Zpq]

def lbt_opt_scale_factor(X, N):
    
    s_factors = np.linspace(1,2, 101)
    CR_max = 0
    sf_max = 0
    for sf in s_factors:
        step_opt, CR, _ = lbt_opt_step(X, N, sf)
        
        if CR > CR_max:
            CR_max = CR
            sf_max = sf
            
    return sf_max

def nlevdwt(X, n):
    Xp = X.copy()
    m= Xp.shape[0]
    Y=dwt(Xp)
    for i in range(n-1):
        m = m//2
        # DTW on first sub-image
        Y[:m,:m] = dwt(Y[:m,:m])
    return Y

def nlevidwt(Y, n):
    Yp = Y.copy()
    m = Yp.shape[0]
    # n layer iDTW, used by n = {2...n}
    for e in range(n-1, 0, -1):
        i = m//2**e
        Yp[:i,:i] = idwt(Yp[:i,:i])
    # Final iDTW, used by final layer n=1
    Xr = idwt(Yp)
    return Xr

def quantdwt(Y, dwtstep, rise=0.5):

    Yp = Y.copy()
    m = Yp.shape[0]
    
    ks, cols = dwtstep.shape
    
    # Number of DWT levels (n)
    levels = cols - 1
    
    # Entropy matrix 
    dwtent = np.zeros_like(dwtstep)
    dwtstep = dwtstep
    
    # Iterate through the levels backwards to reconstruct smaller images first!
    for l in range(levels, -1, -1):
        for k in range(ks):
            if k == 0 and l == levels:
                i = m//2**l
                Xq = quantise(Yp[:i,:i], dwtstep[0, l], rise1 = rise)
                Yp[:i,:i] = Xq
                #print("Xq", Xq.shape)
                dwtent[0,l] = img_size(Xq)
                
            if k == 0 and l < levels:
                i = m//2**(l+1)
                top_right = quantise(Yp[:i,i:2*i], dwtstep[k, l], rise1 = rise)
                Yp[:i,i:2*i] = top_right
                #print("top_right", top_right.shape)
                dwtent[k,l] = img_size(top_right)
                
            if k == 1 and l < levels:
                i = m//2**(l+1)
                bottom_left = quantise(Yp[i:2*i,:i], dwtstep[k, l], rise1 = rise)
                Yp[i:2*i,:i] = bottom_left
                #print("bottom_left", bottom_left.shape)
                dwtent[k,l] = img_size(bottom_left)
                
            if k == 2 and l < levels:
                i = m//2**(l+1)
                bottom_right = quantise(Yp[i:2*i,i:2*i], dwtstep[k, l], rise1 = rise)
                Yp[i:2*i,i:2*i] = bottom_right
                #print("bottom_right", bottom_right.shape)
                dwtent[k,l] = img_size(bottom_right)
                
    Yq = Yp
    
    return Yq, dwtent

def dwt_opt_step(X, n):
    """Optimisation function. Returns both the optimal step size and its CR"""
    
    # Reference Scheme
    Xq = quantise(X, 17)
    ref_size = img_size(Xq)
    goal = np.std(X - Xq)
    Xp = X.copy()
    
    # n-level DWT
    Y = nlevdwt(Xp, n)
    
    # Constant steps for each layer (mse has different ratios_matrix)
    step_matrix = np.ones(3*(n+1)).reshape((3, n+1))
    
    def error_difference(steps):
        
        # Quantise layer
        Yq, *_ = quantdwt(Y, step_matrix*steps)
        
        # Reconstruct (iDWT)
        Zq = nlevidwt(Yq, n)
        
        current = np.std(X - Zq)
        
        error = np.abs(goal-current)

        return error

    res = optimize.minimize_scalar(error_difference, bounds=(0, 256))
    
    step_opt = res.x
    error_opt = res.fun
    
    Yq_opt, entropy_matrix = quantdwt(Y, step_matrix*step_opt)
    size = np.sum(entropy_matrix)
    CR = round(ref_size/size, 4)  
    
    #Reconstruct optimised image
    Zq_opt = nlevidwt(Yq_opt, n)
    
    return [step_opt, CR, Zq_opt]

def mse(Y, n):

    Yp = Y.copy()
    m = Yp.shape[0]
    
    # Number of DWT levels (n)
    levels = n 
    
    # energy matrix 
    mse_energy = np.zeros(3*(n+1)).reshape((3, n+1))
    
    # Iterate through the levels backwards to reconstruct smaller images first!
    for l in range(levels, -1, -1):
        for k in range(3):
            if k == 0 and l == levels:
                i = m//2**l
                mid = i//2
                
                Yp = Y.copy()
                Yp[mid, mid] = 100
                
                Zp = nlevidwt(Yp, n)
                #fig, ax = plt.subplots()
                #plot_image(Yp, ax=ax)
                #ax.set(title="Xq");
                e = np.sum(Zp**2)**0.5
                #print(f"{l} level", e)
                mse_energy[0,l] = e
                
            if k == 0 and l < levels:
                i = m//2**(l+1)
                midx = i//2
                midy = (i+2*i)//2
                
                Yp = Y.copy()
                Yp[midx, midy] = 100
                #fig, ax = plt.subplots()
                #plot_image(Yp, ax=ax)
                #ax.set(title="top_right")
                Zp = nlevidwt(Yp, n)
                e = np.sum(Zp**2)**0.5
                #print("top_right", e)
                mse_energy[k,l] = e
                
            if k == 1 and l < levels:
                i = m//2**(l+1)
                midx = (i+2*i)//2
                midy = i//2
                
                Yp = Y.copy()
                Yp[midx, midy] = 100
                #fig, ax = plt.subplots()
                #plot_image(Yp, ax=ax)
                #ax.set(title="bottom_left")
                Zp = nlevidwt(Yp, n)
                e = np.sum(Zp**2)**0.5
                #print("bottom_left", e)
                mse_energy[k,l] = e
                
            if k == 2 and l < levels:
                i = m//2**(l+1)
                mid = (i+2*i)//2
                
                Yp = Y.copy()
                Yp[mid, mid] = 100
                #fig, ax = plt.subplots()
                #plot_image(Yp, ax=ax)
                #ax.set(title="bottom_right")
                Zp = nlevidwt(Yp, n)
                e = np.sum(Zp**2)**0.5
                #print("bottom_right", e)
                mse_energy[k,l] = e
    
    #ratios
    mse_ratios = mse_energy.copy()
    y0_energy = mse_ratios[0,n]
    
    for j in range(0, n+1):
        if j < n-1:
            mse_ratios[:,j] = 1/(mse_ratios[:,j]/y0_energy)
        if j == n-1:
            mse_ratios[:,j] = 1/(mse_ratios[:,j]/y0_energy)
        if j == n:
            mse_ratios[0,j] = 1/(mse_ratios[0,j]/y0_energy)
     
    return mse_ratios

def dwt_opt_step_mse(X, n):
    """Optimisation function. Returns both the optimal step size and its error"""
    
    Xq = quantise(X, 17)
    ref_size = img_size(Xq)
    goal = np.std(X - Xq)
    Xp = X.copy()
    
    # n-level DWT
    Y = nlevdwt(Xp, n)
    A = np.zeros_like(X)
    mse_ratios = mse(nlevdwt(A, n), n)

    def error_difference(steps):
        
        # Quantise layer
        Yq, _ = quantdwt(Y, mse_ratios*steps)
        
        # Reconstruct (iDWT)
        Zq = nlevidwt(Yq, n)
        
        current = np.std(X - Zq)
        
        error = np.abs(goal-current)

        return error

    res = optimize.minimize_scalar(error_difference, bounds=(0, 256), method='bounded')
    step_opt = res.x
    error_opt = res.fun
    Yq_opt, entropy_matrix = quantdwt(Y, step_opt*mse_ratios)
    
    size = np.sum(entropy_matrix)
    CR = round(ref_size/size, 4)
    
    #Reconstruct optimised image
    Zq_opt = nlevidwt(Yq_opt, n)
    
    return [step_opt, CR, Zq_opt]

