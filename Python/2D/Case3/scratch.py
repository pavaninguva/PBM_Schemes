import numpy as np
import matplotlib.pyplot as plt
#For exact scheme
from scipy.integrate import quad
from functools import partial

def g1fun(x,y):
    g1 = 0.25 + (x*y)
    return g1

def g2fun(y):
    g1 = 0.5 + 0.25*(x+y)
    return g1

def recip(x,y,fun):
        f = 1/(fun(x,y))
        return f

# def x_transfun(x,y):
#     result = np.array(list(map(partial(quad,(lambda x:g1fun(x,y)),0.0),x)))[:,0]
#     return result

def x_transfun(x,y):
        result = np.array(list(map(partial(quad,(lambda x:recip(x,y,g1fun)),0.0),x)))[:,0]
        return result


y = 0.0
x = np.linspace(0.0,2.0,20)

def xtilde_fun(x,y):
    f = np.log(4*x*y + 1)/y
    return f

# print(xtilde_fun(x,y))

fig1 = plt.figure(num=1)
plt.plot(x,x_transfun(x,y))
# plt.plot(x,xtilde_fun(x,y))

plt.show()
