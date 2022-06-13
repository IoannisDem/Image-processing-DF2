#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

