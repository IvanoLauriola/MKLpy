import numpy as np
import mpmath as mp

def UCI(data): #Data: UCI file
    '''Import an UCI format file'''
    f = open('datasets/'+data)
    lines = [line.split() for line in f.readlines()]
    Y = [float(line[0]) for line in lines]
    row = []
    col = []
    val = []
    for (r,line) in enumerate(lines):
        line = line[1:]
        for z in line:
            (i,x)=z.split(":")
            (i,x)=(int(i),float(x))
            row.append(r)
            col.append(i-1)
            val.append(x)
     
    nf = max(col)+1
    nr = max(row)+1
    X = [[0.0 for j in range(nf)] for i in range(nr)]
    for i in range(len(row)):
        X[row[i]][col[i]]=val[i]
    return np.array(X),np.array(Y)


def bignom(n,k):
	if n < k: return mp.mpf(0.0)
	res = mp.mpf(1.0)
	for i in range(0,k):
		res = res * mp.mpf(float(n-i)/(k-i))
	return res