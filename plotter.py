# -*- coding: utf-8 -*-
"""
@author: Carlos
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np


def plotter(x,**kwargs):
    """
    x: lista de arrays a graficar.
    ejes: opcion para renombrar los ejes.
    size: ajusta el tamaño de los puntos individuales.
    color: puede ser un color o un colormap de matplotlib.
    https://matplotlib.org/stable/users/explain/colors/colormaps.html
    """
        
    a,b,c = 'Eje X','Eje Y','Eje Z'
    if 'ejes' in kwargs:
        a,b,c = kwargs['ejes']
        
    if 'size' not in kwargs:
        kwargs['size'] = [1]*len(x)
    
    if type(x) != list:
        x = [x]
    
    size = kwargs['size']
    fig = plt.figure(num = 'phase',figsize=(8,4.5),dpi=120)
    ax = plt.axes(projection='3d')
    
    for j in range(len(x)):
        t = x[j]
        if 'color' in kwargs:
            if isinstance(kwargs['color'][j],str) and kwargs['color'][j] in mpl.colormaps:
                colormap = mpl.colormaps[kwargs['color'][j]]
                tempo = np.linspace(0,1,num=t.shape[0]) 
                color = colormap(tempo)
            elif isinstance(kwargs['color'][j],str):
                color = kwargs['color'][j]
            else:
                color = np.atleast_2d(kwargs['color'][j])
            ax.scatter(t[:,0],t[:,1],t[:,2], c=color, s=size[j])
        else:
            tempo = [0.1]*t.shape[0]
            ax.scatter(t[:,0],t[:,1],t[:,2], s=size[j])
            
    
    ax.set_xlabel(a)
    ax.set_ylabel(b)
    ax.set_zlabel(c)
    ax.zaxis.set_rotate_label(False)
            