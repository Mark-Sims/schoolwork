import numpy as np
import math as m

def pdf(x, u, sigma_squared=float(.01)):
    a = float(1)/(m.sqrt(2*m.pi*sigma_squared)) 
    top = (x-u)**2
    bottom = (float(2)*(sigma_squared))
    print top
    print bottom
    b = float(-1) * top / bottom
    print b
    b = np.exp(b)
    print b
    return b

print pdf(float(11), float(5.8))
