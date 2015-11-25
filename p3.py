from __future__ import division
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

def ajuste(a,b,x):
    return a*x+b


data= np.loadtxt("data/R9Q.dat", usecols=(80,81,82,83))
bandai= data[:,0]
errori= data[:,1]
bandaz= data[:,2]
errorz= data[:,3]
linea= np.polyfit(bandai,bandaz,1)
