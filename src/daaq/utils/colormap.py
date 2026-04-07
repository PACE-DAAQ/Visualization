# utils/colormap.py
from matplotlib.colors import LinearSegmentedColormap

white_gist_earth = LinearSegmentedColormap.from_list('white_gist_earth', [
    (0,     (1,        1,        1       )),
    (1e-20, (0.965882, 0.915975, 0.913378)),
    (0.2,   (0.772885, 0.646409, 0.444171)),
    (0.4,   (0.568932, 0.677541, 0.340330)),
    (0.6,   (0.249216, 0.576471, 0.342046)),
    (0.8,   (0.143740, 0.396564, 0.488306)),
    (1,     (0.013067, 0.000000, 0.348089)),
    ], N=256)

def setup_cmap(name, valuelst, idxlst):
    #
    # Set colormap through NCL colormap and index
    #
    import os, platform
    from pathlib import Path
    import matplotlib.colors as mpcrs
    import numpy as np
    rootpath=Path(__file__).parent
    nclcmap=str(rootpath.resolve())+'/colormaps'
    
    cmapname=name
    f=open(nclcmap+'/'+cmapname+'.rgb','r')
    a=[]
    for line in f.readlines():
        if ('ncolors' in line):
            clnum=int(line.split('=')[1])
        a.append(line)
    f.close()
    values = [x/(valuelst[-1]-valuelst[0]) for x in valuelst]
    b = a[-clnum:]
    c = []
    if ('MPL' in name or 'GMT' in name):
        for idx in idxlst:
            if (i == 0):
                c.append(tuple(float(y) for y in [1,1,1]))
            elif (i == 1):
                c.append(tuple(float(y) for y in [0,0,0]))
            elif (i == -1):
                c.append(tuple(float(y) for y in [0.5,0.5,0.5]))
            else:
                c.append(tuple(float(y) for y in b[idx-2].split('#', 1)[0].split()))
    else:
        for idx in idxlst:
            if (idx == 0):
                c.append(tuple(float(y)/255. for y in [255, 255, 255]))
            elif (idx == 1):
                c.append(tuple(float(y)/255. for y in [0, 0, 0]))
            elif (idx == -1):
                c.append(tuple(round(float(y)/255., 4) for y in [128, 128, 128]))
            else:
                c.append(tuple(round(float(y)/255., 4) 
                                     for y in b[idx-2].split('#', 1)[0].split()))

    d = LinearSegmentedColormap.from_list(name, c, len(idxlst))
    return c, d