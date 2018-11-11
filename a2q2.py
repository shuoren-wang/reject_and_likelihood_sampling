
# coding: utf-8

# In[156]:


import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import math

rs = genfromtxt('rs_1.csv')


# In[157]:


N = 100000 
rs_rain_prob = np.zeros(N)
rs_upper_prob = np.ones(N)
rs_lower_prob = np.zeros(N)

accepted_count=0
rain_count=0
for i in range(1, N):
    if rs[i]!=-1:
        accepted_count+=1
        if rs[i]==1:
            rain_count+=1
    error = math.sqrt(math.log(2/0.05)/(2*accepted_count))
    if i==N-2:
        print(error)
    rs_rain_prob[i]=rain_count/accepted_count
    if rs_rain_prob[i]+error<1:
        rs_upper_prob[i]=rs_rain_prob[i]+error
    if rs_rain_prob[i]>=error:
        rs_lower_prob[i]=rs_rain_prob[i]-error


# In[158]:


# rs_upper_prob[0]
print(accepted_count)
print(rs[rs!=-1].size)
print(rs_upper_prob[N-2])
print(rs_rain_prob[N-2])


# In[110]:


x= np.arange(0,N)
plt.semilogx(x, rs_rain_prob)
plt.title('Reject Sampling')
plt.axis([1, N, 0, 1])
plt.xlabel('Number of Samples, N')
plt.ylabel('P(r|s,w)')
plt.show()
plt.gcf().clear()


# In[159]:


#part b
fig, ax = plt.subplots()
ax.semilogx(x, rs_upper_prob, label='upper bound')
ax.semilogx(x, rs_rain_prob, label='reject sampling')
ax.semilogx(x, rs_lower_prob, label='lower bound')
ax.legend(loc='upper right', frameon=False)
plt.title('Reject Sampling with Confidence bounds')
ax.axis([1, N, 0, 1])
plt.xlabel('Number of Samples, N')
plt.ylabel('P(r|s,w)')


# In[92]:


# part c
lw= genfromtxt('lw_1.csv', delimiter =",")


# In[23]:


# lw[0,0]


# In[123]:


lw_rain_prob = np.zeros(N)
rain_weight=0
total_weight=0
for i in range(0, N-1):
    weight=lw[i,1]
    if lw[i,0]==1:
        rain_weight+=weight
    total_weight+=weight
    lw_rain_prob[i+1]=rain_weight/total_weight


# In[164]:


x= np.arange(0,N)
plt.semilogx(x, lw_rain_prob)
plt.semilogx(x, rs_rain_prob)
plt.axis([1, N, 0, 0.55])
# plt.title('Likelihood Weighting Algorithm Approximation')
plt.title('Likelihood Weighting vs Reject Sampling')
plt.xlabel('Number of Samples, N')
plt.ylabel('P(r|s,w)')
plt.show()
plt.gcf().clear()


# In[126]:


lw_rain_prob[N-1]

