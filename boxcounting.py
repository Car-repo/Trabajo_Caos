# -*- coding: utf-8 -*-
"""
@author: Carlos
"""
import numpy as np
import tqdm.contrib.concurrent
import verner65
from plotter import plotter

rkv65 = verner65.rkv65
eval_interpol = verner65.eval_interpol

def lorenz(**kwargs):
    """
    Devuelve el campo vectorial de Lorenz.
    """

    if 'param' not in kwargs:
        kwargs['param'] = (10,28,8/3)
    if 'reverse' not in kwargs:
        kwargs['reverse'] = False
    
    p,r,b = kwargs['param']
    
    f = lambda x,t : ((-1)**kwargs['reverse'])* \
            np.array((p*(-x[0]+x[1]),-x[0]*x[2]+r*x[0]-x[1],x[0]*x[1]-b*x[2]))

    return f


def limites(x,l=(.095,.1),**kwargs):
    """
    x: array de datos.
    l: intervalo objetivo. El algoritmo termina cuando 
    l[0]*|x| <= N(eps) <= l[1]*|x|.
    info: muestra informacion de cada paso del bucle.
    initial: inicializacion opcional de eps.
    """
    
    if 'info' not in kwargs:
        kwargs['info'] = False
    if 'initial' not in kwargs:
        kwargs['initial'] = 1
    
    N = len(x)
    eps = kwargs['initial']
    prev_eps = kwargs['initial']
    exceded = False
    
    while True: 
        _,N_eps = caja([x,eps])
        
        if kwargs['info']:
            print(f'{prev_eps}, {eps} | ',end='\r',flush=True)
        
        if l[0]*N <= N_eps and N_eps <= l[1]*N:
            result = eps
            break  
        
        # Reduce el epsilon        
        elif N_eps < l[0]*N:       
            if exceded:
                prev_eps = eps
                eps *= np.sqrt(exceded_eps/eps)
            else:
                prev_eps = eps
                eps *= .5
            exceded = False
            
        # Aumenta el epsilon
        else:
            if eps == prev_eps:
                eps *= 2
                prev_eps *= 2
            else:
                exceded = True
                exceded_eps = eps
                eps *= np.sqrt(prev_eps/eps)
            
    return eps


def caja(arg):
    """
    arg 0: array de datos.
    arg 1: epsilon.
    """
    indexes = np.floor_divide(arg[0],arg[1]).astype('int64')
    
    return [arg[1],len(np.unique(indexes,axis=0))]
    

def generacion(arg):
    """
    arg 0: intervalo [0,T].
    arg 1: condicion inicial.
    arg 2: periodo transitorio a descartar, [0, arg 2].
    """
    
    time = arg[0]
    x0 = arg[1]
    tspan = np.linspace(arg[2],time[1],int(1e6))

    sol = rkv65(lorenz(),time,x0,utol=(1e-14,1e-15),dense=1)
    y = eval_interpol(tspan,sol)

    return y



run = 0

if __name__ == "__main__" and run:
    """
     Nota: dependiendo del procesador y la memoria RAM, habria que ajustar el
     numero de procesos, especialmente en la parte de box-counting.
    """
    
    # Parametros iniciales
    ls = [-1,0,1]
    init = [[a,b,c] for c in ls for b in ls for a in ls if a!=0 and b!= 0]
    process_map = tqdm.contrib.concurrent.process_map
    
    
    # Generacion atractor
    args = [[(0,360),x,60] for x in init]
    data = process_map(generacion,args,max_workers=15,chunksize=1)
    data = np.concatenate(data)
    
    eps_inf = limites(data,(.095,.1),initial=.052) # Puede ser lento.
    eps = np.logspace(np.log10(5),np.log10(eps_inf),50)
    
    # Procesamiento en paralelo
    args = [[data,v] for v in eps]
    result = process_map(caja,args,max_workers=8,chunksize=1)
    result = np.asarray(result)
    
    log_res = np.log10(result)*(-1,1)
    dim = np.polyfit(log_res[:,0],log_res[:,1],1)[0]
        
    del ls,init,eps_inf,eps,args,data
     
del run  
  

