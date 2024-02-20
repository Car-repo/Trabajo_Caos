# -*- coding: utf-8 -*-
"""
@author: Carlos
"""
import numpy as np
import tqdm.contrib.concurrent
from verner65 import rkv65
from plotter import plotter


def tarea1(args):
    """
    Generacion de la malla del elipsoide. La funcion esta pensada para ejecutarse
    en paralelo.
    args 0: alpha.
    args 1: numero de puntos.
    args 2: coeficientes de la parametrizacion.
    args 3: vector traslacion
    args 4,5:  
    """
    
    a = args[0]
    granularity = args[1]
    c1,c2,c3 = args[2]
    center = args[3]
    _,r,b = args[4]
    filtro = args[5]
    beta = -np.linspace(0,np.pi/2,granularity,endpoint=False)
    
    l = np.array([(c1*np.cos(a)*np.cos(b),c2*np.sin(a)*np.cos(b),c2*np.sin(b)) 
                  for b in beta])
    
    l += np.array(center)   
    if filtro:       
        l = np.array([t for t in l if r*t[0]**2+t[1]**2+b*(t[2]-r)**2<b*r**2])
    
    return l
        

def tarea2(args):
    """
    Integracion del flujo sobre la region del elipsoide.
    args 0: intervalo [0,T].
    args 1: condicion inicial.
    args 2: parametros del campo vectorial.
    args 3: tolerancia (atol,rtol).
    """
    from boxcounting import lorenz
    
    time = args[0]
    x0 = args[1]
    p,r,b = args[2]
    tol = args[3]
    
    event = lambda x: r*x[0]**2+x[1]**2+b*(x[2]-r)**2-b*r**2>0
    value = lambda y: r*y[0]**2+p*y[1]**2+p*(y[2]-2*r)**2
    
    x = rkv65(lorenz(param=(p,r,b)),time,x0,utol=tol,event=event)['x']
    
    val = np.apply_along_axis(value,1,x)
    
    return x,val


run = 0

if __name__ == '__main__' and run:
    
    
    # Inicializacion
    p,r,b = (10,28,8/3)
    granularity = 4000
    alpha = np.linspace(0,np.pi,granularity)
    tol = (1e-14,1e-15)
    process_map = tqdm.contrib.concurrent.process_map
    
    d = np.sqrt(33450+2/3)-np.sqrt(10*56**2)
    v = 31360+1/9*d**2+2/3*d*np.sqrt(10*56**2)
    coeff = np.sqrt(v)*np.sqrt([1/r,1/p,1/p])

    del d
    
    # Genera puntos de la interseccion
    args = [[a,granularity,coeff,[0,0,2*r],[p,r,b],1] for a in alpha] 
    puntos = process_map(tarea1,args,max_workers=15,chunksize=4)
    puntos = np.concatenate([a for a in puntos if a.shape[0]>0])
    
    # Integracion de la region
    args = [[[0,3],x,(p,r,b),tol] for x in puntos]
    result = process_map(tarea2,args,max_workers=15,chunksize=16)
    result = np.asarray(result,dtype='object')

    del p,r,b,granularity,tol,alpha,v,coeff,args

del run

