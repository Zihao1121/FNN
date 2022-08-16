#!/usr/bin/env python
# coding: utf-8

# In[83]:


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


# In[84]:


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


# In[85]:


def get_w1w2w3(n1,n2):

    w1=np.random.random((n1,5))
    w2=np.random.random((n2,n1+1))
    w3=np.random.random((1,n2+1))
    return w1,w2,w3


# In[86]:


def get_y(z):
    y=1/(1+np.exp(-z))
    return y


# In[87]:


def get_prediction(z):
    y=get_y(z)
    if(y>=0.5):
        return 1
    else:
        return 0


# In[88]:


def misclassification(t,y):
    err=0
    for i in range(len(t)):
        if(y[i]!=t[i]):
            err=err+1
    return err


# In[89]:


def g_z(z):
    N=len(z)
    g=z
    for i in range(N):
        if (z[i]>=0):
            g[i]=1
        else:
            g[i]=0
    return g


# In[90]:


def iteration(x,t,w1,w2,w3,alpha):

    alpha=0.005
    lambd=0.01
    N=len(x)
    z1,h1=hidden_layer(x,w1)
    z2,h2=hidden_layer(h1,w2)
    z3,h3=hidden_layer(h2,w3)

    g2=g_z(z2)
    g1=g_z(z1)

    w_3=w3[:,1:]
    w_2=w2[:,1:]
    w_1=w1[:,1:]
    zero_w1=np.insert(w_1, 0,0, axis=1)
    zero_w2=np.insert(w_2, 0,0, axis=1)
    zero_w3=np.insert(w_3, 0,0, axis=1)

    omega1=np.square(w_1)
    omega2=np.square(w_2)
    omega3=np.square(w_3)


    omega1=np.sum(omega1,axis=1)
    omega1=np.reshape(omega1,(len(omega1),1))
    omega2=np.sum(omega2,axis=1)
    omega2=np.reshape(omega2,(len(omega2),1))
    omega3=np.sum(omega3,axis=1)
    omega3=np.reshape(omega3,(len(omega3),1))


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


# In[91]:


def iteration_bonus(x,t,w1,w2,w3,alpha):

    alpha=0.005
    lambd=0.03
    N=len(x)
    z1,h1=hidden_layer(x,w1)
    z2,h2=hidden_layer(h1,w2)
    z3,h3=hidden_layer(h2,w3)

    g2=g_z(z2)
    g1=g_z(z1)

    w_3=w3[:,1:]
    w_2=w2[:,1:]
    w_1=w1[:,1:]
    zero_w1=np.insert(w_1, 0,0, axis=1)
    zero_w2=np.insert(w_2, 0,0, axis=1)
    zero_w3=np.insert(w_3, 0,0, axis=1)

    omega1=np.square(w_1)
    omega2=np.square(w_2)
    omega3=np.square(w_3)


    omega1=np.sum(omega1,axis=1)
    omega1=np.reshape(omega1,(len(omega1),1))
    omega2=np.sum(omega2,axis=1)
    omega2=np.reshape(omega2,(len(omega2),1))
    omega3=np.sum(omega3,axis=1)
    omega3=np.reshape(omega3,(len(omega3),1))


    y=get_y(z3)
    h2t=h2.T
    h2t_dummy=np.insert(h2t,0,1,axis=1)
    h1t=h1.T
    h1t_dummy=np.insert(h1t,0,1,axis=1)
    xt=np.reshape(x,(1,N))
    xt_dummy=np.insert(xt,0,1,axis=1)

    dev_j_z3=-t+y+lambd*omega3
    dev_j_w3=np.dot(dev_j_z3,h2t_dummy)+2*lambd*zero_w3

    dev_j_z2=np.multiply(g2,np.dot(w_3.T,dev_j_z3))+lambd*omega2
    dev_j_w2=np.dot(dev_j_z2,h1t_dummy)+2*lambd*zero_w2

    dev_j_z1=np.multiply(g1,np.dot(w_2.T,dev_j_z2))+lambd*omega1
    dev_j_w1=np.dot(dev_j_z1,xt_dummy)+2*lambd*zero_w1

    w1=w1-alpha*dev_j_w1
    w2=w2-alpha*dev_j_w2
    w3=w3-alpha*dev_j_w3
    cost=t * np.logaddexp(0,-z3) + (1-t) * np.logaddexp(0,z3)
    return w1,w2,w3,cost


# In[92]:


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


# In[93]:


def SGD(X_train,t_train,n1,n2,alpha,w1,w2,w3):

    N=len(X_train)
    cost=0
    cost = np.zeros(N)
    val=np.zeros(N)
    threshold=0.01
    for i in range (N):
        x=X_train[i]
        t=t_train[i]
        w1,w2,w3,cost[i]=iteration(x,t,w1,w2,w3,alpha)
        cost_mean=sum(cost[0:i])/(i+1)
        #val[i]=error_calculation(w1,w2,w3,X_val,t_val)[1]

    
    final_cost=cost[i]
           
    return w1,w2,w3,final_cost


# In[94]:


def SGD_val(X_train,t_train,n1,n2,alpha,w1,w2,w3):

    N=len(X_train)
    cost=0
    cost = np.zeros(N)
    val=np.zeros(N)
    threshold=0.01
    for i in range (N):
        x=X_train[i]
        t=t_train[i]
        w1,w2,w3,cost[i]=iteration(x,t,w1,w2,w3,alpha)
        cost_mean=sum(cost[0:i])/(i+1)
        val[i]=error_calculation(w1,w2,w3,X_val,t_val)[1]

    
    final_cost=cost[i]
           
    return w1,w2,w3,cost,val


# In[95]:


def SGD_bonus(X_train,t_train,n1,n2,alpha,w1,w2,w3):

    N=len(X_train)
    cost=0
    cost = np.zeros(N)
    val=np.zeros(N)
    threshold=0.01
    for i in range (N):
        x=X_train[i]
        t=t_train[i]
        w1,w2,w3,cost[i]=iteration_bonus(x,t,w1,w2,w3,alpha)
        cost_mean=sum(cost[0:i])/(i+1)
        #val[i]=error_calculation(w1,w2,w3,X_val,t_val)[1]

    
    final_cost=cost[i]
        



    
    return w1,w2,w3,final_cost


# In[96]:


def SGD_bonus_val(X_train,t_train,n1,n2,alpha,w1,w2,w3):

    N=len(X_train)
    cost=0
    cost = np.zeros(N)
    val=np.zeros(N)
    threshold=0.01
    for i in range (N):
        x=X_train[i]
        t=t_train[i]
        w1,w2,w3,cost[i]=iteration_bonus(x,t,w1,w2,w3,alpha)
        cost_mean=sum(cost[0:i])/(i+1)
        val[i]=error_calculation(w1,w2,w3,X_val,t_val)[1]

    
    final_cost=cost[i]
        



    
    return w1,w2,w3,cost,val


# In[97]:


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
        
        


# In[98]:


def epoch_bonus(Len,n1,n2,X_val,t_val,X_train,t_train):
    alpha=0.005
    min_val_entropy=10000
    
    for j in range(5):
        w1_init,w2_init,w3_init=get_w1w2w3(n1,n2)
        for i in range(Len):
            index = [i for i in range(len(X_train))]
            np.random.shuffle(index)
            X_train= X_train[index]
            t_train = t_train[index]   
            w1,w2,w3,c=SGD_bonus(X_train,t_train,n1,n2,alpha,w1_init,w2_init,w3_init)
            rate,validation_entropy=error_calculation(w1,w2,w3,X_val,t_val)
            if(validation_entropy<min_val_entropy):
                min_val_entropy=validation_entropy
                w1_out,w2_out,w3_out,c=SGD_bonus(X_train,t_train,n1,n2,alpha,w1_init,w2_init,w3_init)
                min_rate=rate
    return min_val_entropy,c,min_rate,w1_out,w2_out,w3_out
        


# In[17]:


w1_init,w2_init,w3_init=get_w1w2w3(5,8)
index = [i for i in range(len(X_train))]
np.random.shuffle(index)
X_train= X_train[index]
t_train = t_train[index] 


# In[81]:


#without weight decay
index = [i for i in range(len(X_train))]
np.random.shuffle(index)
X_train= X_train[index]
t_train = t_train[index] 
w1,w2,w3,c,val=SGD_val(X_train,t_train,5,8,0.005,w1_init,w2_init,w3_init)

lin = np.linspace(1,len(c),len(c))
plt.scatter(lin, c, color = 'blue', label='cross validation err')
lin2 = np.linspace(1,len(val),len(val))
plt.scatter(lin2, val, color = 'green', label=' validation err')

error_rate,val_cost=error_calculation(w1,w2,w3,X_val,t_val)
print("error rate=",error_rate,"val_cost=",val_cost)


# In[64]:


# with weight decay
index = [i for i in range(len(X_train))]

w1,w2,w3,c,val=SGD_bonus_val(X_train,t_train,5,8,0.005,w1_init,w2_init,w3_init)

lin = np.linspace(1,len(c),len(c))
plt.scatter(lin, c, color = 'blue', label='cross validation err')
lin2 = np.linspace(1,len(val),len(val))
plt.scatter(lin2, val, color = 'green', label=' validation err')

error_rate,val_cost=error_calculation(w1,w2,w3,X_val,t_val)
print("error rate=",error_rate,"val_cost=",val_cost)


# In[82]:


error_rate,val_cost=error_calculation(w1,w2,w3,X_val,t_val)
print("error rate=",error_rate,"val_cost=",val_cost)


# In[21]:


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


# In[22]:


min_validation=10000
for n1 in range(4,5):
    for n2 in range (2,3):
        print("When n1=",n1,"n2=",n2)
        val,c,rate,w1,w2,w3=epoch(10,n1,n2,X_val,t_val,X_train,t_train)
        print("min_validation_entropy=",val)
        print("At this time,traning_entropy=",c)
        print("At this time,misclassification rate=",rate)
        print("\n")
        if(val<min_validation):
            min_validation=val
            n1_best,n2_best=n1,n2
            w1_best,w2_best,w3_best=w1,w2,w3


# In[23]:


print("weight 1 for the final model is\n",w1_best,"\n")
print("weight 2 for the final model is\n",w2_best,"\n")
print("weight 3 for the final model is\n",w3_best,"\n")


# In[24]:


w1,w2,w3,c,val=SGD_val(X_train,t_train,4,2,0.005,w1_init,w2_init,w3_init)

lin = np.linspace(1,len(c),len(c))
plt.scatter(lin, c, color = 'blue', label='cross validation err')
lin2 = np.linspace(1,len(val),len(val))
plt.scatter(lin2, val, color = 'green', label=' validation err')


# In[25]:


rate,validation=error_calculation(w1,w2,w3,X_val,t_val)
print("rate of misclassification for the final model=",rate)
print("final validation error for the final model=",validation)


# In[26]:


rate,validation=error_calculation(w1,w2,w3,X_test,t_test)
print("rate of misclassification on test set with the final model=",rate)
print("number of misclassification on test set with the final model=",rate*len(X_test))
print("final test entropy with the final model=",validation)


# In[27]:


min_validation=10000
for n1 in range(4,5):
    for n2 in range (6,7):
        print("When n1=",n1,"n2=",n2)
        val,c,rate,w1,w2,w3=epoch_bonus(10,n1,n2,X_val,t_val,X_train,t_train)
        print("min_validation_entropy=",val)
        print("At this time,traning_entropy=",c)
        print("At this time,misclassification rate=",rate)
        print("\n")
        if(val<min_validation):
            min_validation=val
            n1_best,n2_best=n1,n2
            w1_best,w2_best,w3_best=w1,w2,w3


# In[28]:


print("weight 1 for the final model is\n",w1_best,"\n")
print("weight 2 for the final model is\n",w2_best,"\n")
print("weight 3 for the final model is\n",w3_best,"\n")


# In[29]:


w1,w2,w3=get_w1w2w3(4,6)
index = [i for i in range(len(X_train))]
np.random.shuffle(index)
X_train= X_train[index]
t_train =t_train[index] 
w1,w2,w3,c,val=SGD_bonus_val(X_train,t_train,4,2,0.005,w1_init,w2_init,w3_init)

lin = np.linspace(1,len(c),len(c))
plt.scatter(lin, c, color = 'blue', label='cross validation err')
lin2 = np.linspace(1,len(val),len(val))
plt.scatter(lin2, val, color = 'green', label=' validation err')


# In[30]:


rate,validation=error_calculation(w1,w2,w3,X_val,t_val)
print("rate of misclassification for the final model=",rate)
print("final validation error for the final model=",validation)


# In[31]:


rate,validation=error_calculation(w1,w2,w3,X_test,t_test)
print("rate of misclassification on test set with the final model=",rate)
print("number of misclassification on test set with the final model=",rate*len(X_test))
print("final test entropy with the final model=",validation)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:






# In[ ]:





# In[ ]:





# In[ ]:



    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




