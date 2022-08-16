#!/usr/bin/env python
# coding: utf-8

# In[126]:


def hidden_layer(x,w):
    N=len(x)
    #N,M=x.shape
    x=x.reshape(N,1)
    dummy_row=np.ones(1) 
    #dummy_row=np.ones(M)  
    x=np.insert(x,0,dummy_row,axis=0)
    z=w.dot(x)
    h=np.maximum(z, 0) # We use ReLU
    return z,h


# In[127]:


def get_w1w2w3(n1,n2):

    w1=np.random.random((n1,5))
    w2=np.random.random((n2,n1+1))
    w3=np.random.random((1,n2+1))
    return w1,w2,w3


# In[128]:


def get_y(z):
    y=1/(1+np.exp(-z))
    return y


# In[129]:


def get_prediction(z):
    y=get_y(z)
    if(y>=0.5):
        return 1
    else:
        return 0


# In[130]:


def misclassification(t,y):
    err=0
    for i in range(len(t)):
        if(y[i]!=t[i]):
            err=err+1
    return err


# In[131]:


def g_z(z):
    N=len(z)
    g=z
    for i in range(N):
        if (z[i]>=0):
            g[i]=1
        else:
            g[i]=0
    return g


# In[132]:


def iteration(x,t,w1,w2,w3,alpha):
    N=len(x)
    z1,h1=hidden_layer(x,w1)
    z2,h2=hidden_layer(h1,w2)
    z3,h3=hidden_layer(h2,w3)

    g2=g_z(z2)
    g1=g_z(z1)

    w_3=w3[:,1:]
    w_2=w2[:,1:]
    w_1=w1[:,1:]
    
    

    y=get_y(z3)
    h2t=h2.T
    h2t_dummy=np.insert(h2t,0,1,axis=1)
    h1t=h1.T
    h1t_dummy=np.insert(h1t,0,1,axis=1)
    xt=np.reshape(x,(1,N))
    xt_dummy=np.insert(xt,0,1,axis=1)

    dev_j_z3=-t+y
    dev_j_w3=np.dot(dev_j_z3,h2t_dummy)

    dev_j_z2=np.multiply(g2,np.dot(w_3.T,dev_j_z3))
    dev_j_w2=np.dot(dev_j_z2,h1t_dummy)

    dev_j_z1=np.multiply(g1,np.dot(w_2.T,dev_j_z2))
    dev_j_w1=np.dot(dev_j_z1,xt_dummy)

    w1=w1-alpha*dev_j_w1
    w2=w2-alpha*dev_j_w2
    w3=w3-alpha*dev_j_w3
    cost=t * np.logaddexp(0,-z3) + (1-t) * np.logaddexp(0,z3)
    return w1,w2,w3,cost


# In[165]:


def error_calculation(w1,w2,w3,X_valid,t_valid):
    N=len(X_valid)
    cost=np.zeros(N)
    predict=np.zeros(N)
    for i in range (N):
        x=X_valid[i]
        t=t_valid[i]
        z1,h1=hidden_layer(x,w1)
        z2,h2=hidden_layer(h1,w2)
        z3,h3=hidden_layer(h2,w3)
        cost[i]=t * np.logaddexp(0,-z3) + (1-t) * np.logaddexp(0,z3)
        predict[i]=get_prediction(z3)
    error=misclassification(t_valid,predict)
    rate=error/N
    cost_mean=sum(cost)/N
    return rate,cost_mean


# In[436]:


def SGD(X_train,t_train,n1,n2,alpha,w1,w2,w3):

    N=len(X_train)
    cost=0
    cost = np.zeros(N)
    threshold=0.01
    for i in range (N):
        x=X_train[i]
        t=t_train[i]
        w1,w2,w3,cost[i]=iteration(x,t,w1,w2,w3,alpha)
        cost_mean=sum(cost[0:i])/(i+1)

    
    final_cost=cost[i]
        



    
    return w1,w2,w3,final_cost


# In[372]:


def early_stop(i,end,mean,array,threshold):
    if(i==end):
        flag=1
        return flag
    if((array[i]-mean)<threshold):
        return early_stop(i+1,end,mean,array,threshold)
    else:
        flag=0
        return flag


# In[373]:


def epoch(Len,n1,n2,X_val,t_val,X_train,t_train):
    alpha=0.005
    min_val_entropy=10000
    
    for j in range(5):
        w1_init,w2_init,w3_init=get_w1w2w3(n1,n2)
        for i in range(Len):
            index = [i for i in range(len(X_train))]
            np.random.shuffle(index)
            X_train= X_train[index]
            t_train = t_train[index]   
            w1,w2,w3,c=SGD(X_train,t_train,n1,n2,alpha,w1_init,w2_init,w3_init)
            rate,validation_entropy=error_calculation(w1,w2,w3,X_val,t_val)
            if(validation_entropy<min_val_entropy):
                min_val_entropy=validation_entropy
                w1_out,w2_out,w3_out,c=SGD(X_train,t_train,n1,n2,alpha,w1_init,w2_init,w3_init)
                min_rate=rate
    return min_val_entropy,c,min_rate,w1_out,w2_out,w3_out
        
        


# In[437]:


min_validation=10000
for n1 in range(4,7):
    for n2 in range (1,10):
        print("When n1=",n1,"n2=",n2)
        val,c,rate,w1,w2,w3=epoch(30,n1,n2,X_val,t_val,X_train,t_train)
        print("min_validation_entropy=",val)
        print("At this time,traning_entropy=",c)
        print("At this time,misclassification rate=",rate)
        print("\n")
        if(val<min_validation):
            min_validation=val
            n1_best,n2_best=n1,n2
            w1_best,w2_best,w3_best=w1,w2,w3


# In[433]:



w1,w2,w3=get_w1w2w3(5,8)
index = [i for i in range(len(X_train))]
np.random.shuffle(index)
X_train= X_train[index]
t_train = t_train[index] 
w1,w2,w3,c=SGD(X_train,t_train,5,8,0.005,w1,w2,w3)

lin = np.linspace(1,len(c),len(c))
plt.scatter(lin, c, color = 'blue', label='cross validation err')


# In[431]:


error_calculation(w1,w2,w3,X_val,t_val)[0]


# In[313]:


val,c,rate=epoch(5,6,7,X_val,t_val,X_train,t_train)[0:3]
print(val,c,rate)


# In[302]:


#main code
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = np.random.random((3, 2)) - 0.5
N,M=x.shape
dummy_row=np.ones(M)
x=np.insert(x,0,dummy_row,axis=0)
print(x.shape)
print(np.maximum(x, 0))


data = np.loadtxt('data_banknote_authentication.txt',dtype=np.float32,delimiter=',')


X_data=data[:,0:4]
t=data[:,4]


X_train,X_test,t_train,t_test=train_test_split(X_data,t,test_size=0.2,random_state=676)
X_train, X_val, t_train, t_val= train_test_split(X_train, t_train, test_size=0.25, random_state=676)
np.random.seed(676)

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_val=sc.transform(X_val)
N=len(X_train)
print(N)


# In[293]:


a = X_train[1:10]
b = t_train[1:10]
print(a)
print(b)
random.seed(1)
np.random.shuffle(a)
random.seed(1)
np.random.shuffle(b)

 
print(a)
print(b)


# In[271]:


#b = [1, 2,3, 4, 5,6 , 7,8 ,9]
#a = [0,1,0,0,1,1,0,1,1]
a = X_train[1:10]
b = t_train[1:10]
c = list(zip(a, b))
#print(c)
random.Random(100).shuffle(c)
#print(c)
a, b = zip(*c)
print(a)
print(b)


# In[7]:


def get_w1w2w3(n1,n2):

    w1=np.random.random((n1,5))
    w2=np.random.random((n2,n1+1))
    w3=np.random.random((1,n2+1))
    return w1,w2,w3



# In[95]:


w1,w2,w3=get_w1w2w3(7,8)


# In[79]:






# In[77]:


x=X_train.T[]
h=hidden_layer(X_train.T,w1)
print(h.shape)
h2=hidden_layer(h,w2)
print(h2)


# In[ ]:


def get_del_z(w,z_d,z_d-1,t,n):
    if(n==3):
        dev_z=-t+1/(1+math.exp(-z_d))


# In[33]:


def get_y(h,z):
    


# In[31]:


X_data=data[:,0:4]
t=data[:,4]
print(X_data)
print(t)
X_train,X_test,t_train,t_test=train_test_split(X_data,t,test_size=0.2,random_state=676)
X_train, X_val, y_train, y_val= train_test_split(X_train, t_train, test_size=0.25, random_state=676)
np.random.seed(676)


# In[35]:





# In[8]:


random.seed()
np.random.shuffle(X_train)
print(X_train)


# In[264]:


from random import shuffle
import numpy as np
import random
 
a = [[1,2],[3,4]]
b = [[5,6],[7,8]]
 
random.seed(1)
random.shuffle(a)
random.seed(1)
random.shuffle(b)

 
print(a)
print(b)


# In[ ]:


def SGD(X_train,t_train,n1,n2,alpha,w1,w2,w3):

    N=len(X_train)
    cost=0
    cost = np.zeros(N)
    threshold=0.3
    for i in range (N):
        x=X_train[i]
        t=t_train[i]
        w1,w2,w3,cost[i]=iteration(x,t,w1,w2,w3,alpha)
        cost_mean=sum(cost[0:i])/(i+1)
        if(i>50):
            if(early_stop(i-50,i,cost_mean,cost,threshold)==1):
                final_cost=cost[i]
                break
    
    final_cost=cost[i]
        



    
    return w1,w2,w3,final_cost

