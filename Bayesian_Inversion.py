#!/usr/bin/env python
# coding: utf-8

# ## True model is
# ### 0 m to 2000 m - resistivity is 100 ohm-m  
# ### 2000 m to 10000 m - resistivity is 10 ohm-m 
# ### below 10000 m - resistivity is 1000 ohm-m 

# In[19]:


import numpy as np
import math
import cmath
import scipy.stats as st
import matplotlib.pyplot as plt
import scipy.interpolate
import seaborn as sns


# In[20]:


np.random.seed(123)


# In[21]:


mu = 4*math.pi*1E-7
n=3
thicknesses = [2000,8000]
depth=np.cumsum(thicknesses)
frequencies = [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]
resistivities=[100,10,1000]


# In[22]:


# Forward op

def f(resistivities):
    data=[]
    for frequency in frequencies:
        w =  2*math.pi*frequency
        impedances = [1]*n
        impedances[n-1] = cmath.sqrt(w*mu*resistivities[n-1]*1j)
        for j in range(n-2,-1,-1):
            resistivity = resistivities[j]
            thickness = thicknesses[j]
            kj = cmath.sqrt((w * mu * (1/resistivity))*1j)
            Ij = kj * resistivity
            ej = cmath.exp(-2*thickness*kj)
            Zb = impedances[j + 1]
            rj = (Ij - Zb)/(Ij + Zb)
            re = rj*ej
            Zj = Ij * ((1 - re)/(1 + re))
            impedances[j] = Zj
        Z = impedances[0]
        data.append(abs(Z))
    return np.array(data)


# In[23]:


# Probability distributions

def prior(m):
    return st.uniform(0,4000).pdf(m)


def posterior(m):
    fm=f(m)
    d=f(resistivities)
    dcov = np.identity(len(d))
    likelihood=np.exp((-np.matmul((d-fm).T,(d-fm)))/2)
    return np.product(likelihood*prior(m))


# In[26]:


# Bayesian MCMC

def main():
    niter = 10000
    m=np.zeros((niter,n),dtype=float)
    m[0] = np.array([2]*n) #initialization
    counter = 0
    df=[]
    posteriors=[]
    
    for i in range(0, niter-1): 
        m_next = m[i]+np.random.normal(0, 150,n)
        if np.random.random_sample() < min(1, posterior(m_next)/posterior(m[i])):
            m[i+1] = m_next
            if i > 200:      # Burning process
                posteriors.append(posterior(m_next))
                df.append(m_next)
                counter = counter + 1
        else:
            m[i+1] = m[i]

    print("acceptance fraction is ", counter/float(niter))
    index = posteriors.index(max(posteriors))
    print("MAP : ", df[index])
    return df, posteriors
    


# In[27]:


main()


# In[ ]:





# In[ ]:




