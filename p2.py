'''buscando la constante de Hubble a partir de los datos con metodo de supernovas tipo1'''

from __future__ import division
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

def leer(algo):
    '''lee el archivo'''
    data= np.loadtxt(algo, usecols=(1,2))
    return data

def aproximacion1(x,a):
    ''' funcion para minimizar con a el parametro '''
    a= a
    y= a*x # caso modelo de hubble
    return y
def aproximacion2(y,a):
    ''' funcion para minimizar con a el parametro '''
    a= a
    x= y/a # caso modelo de hubble
    return x

def res1 (b,x,y):
    b= b
    aprox= aproximacion1(x,b)
    resta= y- aprox
    return resta

def res2 (b,x,y):
    b= b
    aprox= aproximacion2(y,b)
    resta= x- aprox
    return resta

#usaremos least square para optimizar

def bootstrapt(xx,yy):
    '''usando bootstrapt para buscar el intervalo de confianza'''
    n=len(xx)
    iterando=int(n**2)
    prom= np.zeros(iterando)
    for i in range(iterando):
        random= npr.randint(0,n,size=n)
        xi= xx[random]
        yi= yy[random]
        minimi1= leastsq(res1,h0,args=(xi,yi))
        minimi2= leastsq(res2,h0,args=(yi,xi))
        minimi= (minimi1[0] + minimi2[0])/2 #promedio ambos modelos
        prom[i]=minimi[0]
    stat= np.sort(prom)
    inferior= 0.025
    superior= 0.975
    inf=prom[int(inferior* iterando)]
    sup=prom[int(superior* iterando)]
    return inf, sup
    print "El intervalo de confianza al 95% es: [{}:{}]".format(inf, sup)

algo= "data/SNIa.dat"
data= leer(algo)
d= data[:,1]
v= data[:,0]
h0= 400
aproximando1=leastsq(res1,h0,args=(d,v))
aproximando2=leastsq(res2,h0,args=(d,v))
o=bootstrapt(d,v)

promediando= (aproximando1[0] + aproximando2[0])/2
print (aproximando1[0], aproximando2[0], promediando)
print o

#grafico
graf= np.linspace(0,500,10**6)
aa=aproximacion1(graf,promediando)
print aa
fig= plt.figure()
plt.scatter(d,v)
plt.plot(graf,aa,'b', label= "promedio H_0")
plt.xlabel("distancia [Mpc]")
plt.ylabel("velocidad [km/s]")
plt.title('Datos segunda version')
plt.savefig('2')
plt.show()
