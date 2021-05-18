#!/usr/bin/env python
# coding: utf-8

# In[90]:


import numpy as np
from numpy.random import seed
from numpy.random import randint


# In[91]:


def randomization(n):
    #generate array of random integer numbers ranging from 0 to 100 and size n*1
    A=randint(0,100,size=(n,1))
    return A


# In[92]:


randomization(5)


# In[93]:


def operations(h, w):
    #create matrix of random integers ranging from 0 to 5 and size h*w
    A=np.random.randint(0,5, size=(h, w))
    B=np.random.randint(0,5, size=(h, w))
    #element-wise addition and return result in s
    s=A+B
    print(A)
    print(B)
    return s


# In[94]:


operations(2,4)


# In[95]:


def norm(A, B):
    s=A+B
    return np.linalg.norm(s)


# In[96]:


norm(([1, 0, 3, 0],[4, 1, 1, 1]),([4, 2, 0, 0],[4, 0, 2, 0]))


# In[97]:


def neural_network(inputs, weights):
    w=np.transpose(weights)
    z=np.tanh(np.dot(w,inputs))
    return z


# In[98]:


neural_network(([2,1]),([3,3]))


# In[99]:


def scalar_function(x, y):
    if (x<y):
        return np.dot(x,y)
    return x/y


# In[119]:


def vector_function(x, y):
    return np.vectorize(scalar_function)(x,y)


# In[120]:


scalar_function(3,2)


# In[123]:


vector_function([3,3,3],[2,2,2])


# In[ ]:




