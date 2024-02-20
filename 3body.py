# -*- coding: utf-8 -*-
"""
@author: Carlos
"""
import numpy as np
import verner65
import matplotlib.pyplot as plt
import tqdm.contrib.concurrent
import boxcounting

rkv65 = verner65.rkv65
interpolate = verner65.eval_interpol
caja = boxcounting.caja
limites = boxcounting.limites

x0 = np.array([[.994,0,0,-2.001585106379082523]])
nu = .012277471


def grav(nu=nu):
    """
    Campo vectorial experimentado por el tercer cuerpo.
    """
    
    def f(x,t):
        x = x.reshape(-1)
        c1 = np.power((x[0]+nu)**2+x[1]**2,-1.5)
        c2 = np.power((x[0]+nu-1)**2+x[1]**2,-1.5)
        
        a1 = x[0]+2*x[3]-(1-nu)*(x[0]+nu)*c1-nu*(x[0]+nu-1)*c2
        a2 = x[1]-2*x[2]-(1-nu)*x[1]*c1-nu*x[1]*c2
        
        return np.array([x[2],x[3],a1,a2])

    return f


def tiempo_caos(sols,tol):
    """
    sols: (array x, array y).
    tol: separacion espacial maxima permitida entre x e y.
    """
    
    dist = lambda p: np.sqrt(p[0]**2+p[1]**2)
    x,y = sols
    for j in range(len(x)):
        if dist(x[j]-y[j]) > tol:           
            break
        result = j
        
    return j


def medida(arg):
    """
    arg 0: resultado de rkv65.
    arg 1: extremo del intervalo [0, T_l].
    """
    
    tspan = np.linspace(0,arg[1],int(1e6))
    data = interpolate(tspan,arg[0])

    # Eleccion automatica del rango de tamaños
    eps_sup = limites(data,(.0095,.01),initial=.5)
    eps_inf = limites(data,(.095,.1),initial=.2)

    eps = np.logspace(np.log10(eps_sup),np.log10(eps_inf),10)
    
    result = [0]*len(eps)
    for j in range(len(eps)):
        result[j] = caja([data,eps[j]])

    result = np.asarray(result)

    return result


def tarea(arg):
    """
    arg 0: intervalo [0,T].
    arg 1: parametro de masa.
    arg 2: condicion inicial.
    arg 3: paso inicial de rkv65.
    arg 4: tolerancias.
    """
       
    time = arg[0]
    nu = arg[1]
    x = arg[2]
    step0 = arg[3]
    l,h = arg[4]
    terminate = lambda x: np.sqrt((x[0]+nu-1)**2+x[1]**2)<5e-6
    
    # Integra el PVI dos veces
    sol_l = rkv65(grav(nu),time,x,step=step0,utol=l,dense=True,
                     event=terminate)
    sol_h = rkv65(grav(nu),time,x,step=step0,utol=h,dense=True,
                      event=terminate)
    
    t1,t2 = sol_l['t'],sol_h['t']
    
    # Prepara las soluciones para compararlas directamente
    tspan = np.linspace(time[0],min(t1[-1],t2[-1]),num=int(1e6))
    x = interpolate(tspan,sol_l)
    y = interpolate(tspan,sol_h)
    
    # Encuentra el tiempo de separacion
    t_limit = tiempo_caos((x,y),1e-3)
    
    return sol_h,tspan[t_limit]



run = 0

if __name__ == '__main__' and run:
    
    # Parametros
    initial = np.array([[t,0,0,0] for t in np.linspace(-1e-5,1e-5,50)])+x0
    time = [0,150]    
    l_tol = (1e-10,1e-10)
    h_tol = (1e-14,1e-14)
    step0 = 5e-5
    args = [[time,nu,x,step0,(l_tol,h_tol)] for x in initial]
    process_map = tqdm.contrib.concurrent.process_map
    
    # Estima el punto de divergencia/caos
    temp = process_map(tarea,args,max_workers=15,chunksize=1)
    sol,tl = zip(*temp)
    tl = np.asarray(tl)
    
    # Dimension por cajas de cada solucion
    args = [[sol[j],tl[j]] for j in range(len(initial))]
    raw = process_map(medida,args,max_workers=15,chunksize=1)    
    temp = [np.log10(w)*(-1,1) for w in raw]
    dim = np.asarray([np.polyfit(w[:,0],w[:,1],1) for w in temp])
    
    del initial,time,l_tol,h_tol,step0,args,temp
      
del run    
