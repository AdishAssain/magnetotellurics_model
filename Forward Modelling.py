#!/usr/bin/env python
# coding: utf-8

# ## Forward Modelling  : Magnetotellurics

# In[7]:


import math
import cmath
import numpy as np


# In[11]:


mu = 4*math.pi*1E-7 # Magnetic Permeability
n=3 # Number of layers
thicknesses = [2000,8000]
frequencies = [0.0001,0.001,0.01,0.1,1,10,100,1000,10000]
resistivities=[100,10,1000]


# In[13]:


# Forward modelling operator
# Input : resistivities ,,, Output : Wave impedence at top of layered model

def f(resistivities):
    data=[]
    for frequency in frequencies:
        w =  2*math.pi*frequency
        impedances = [1]*n
        impedances[n-1] = cmath.sqrt(w*mu*resistivities[n-1]*1j) # Zn (halfspace)
        for j in range(n-2,-1,-1):
            resistivity = resistivities[j]
            thickness = thicknesses[j]
            kj = cmath.sqrt((w * mu * (1/resistivity))*1j)       # wave number
            Ij = kj * resistivity                             # intrinsic impedance
            ej = cmath.exp(-2*thickness*kj)                   # exponential factor
            Zb = impedances[j + 1]               # impedance of bottom layer
            rj = (Ij - Zb)/(Ij + Zb)            # reflection coefficient
            re = rj*ej
            Zj = Ij * ((1 - re)/(1 + re))      # j-th impedance
            impedances[j] = Zj
        Z = impedances[0]                      # impedance at top
        apparentresistivity = (abs(Z) * abs(Z))/(mu*w)
        phase = math.atan2(Z.imag, Z.real)
        data.append([frequency,apparentresistivity,phase])
    return np.array(data)


# In[14]:


f(resistivities)


# In[ ]:




