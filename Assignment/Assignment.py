#import libararies here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

fileName = 'Training_Data.csv'
rawTrainingData = pd.read_csv(fileName).values
f = rawTrainingData[:,[6]] 
f = f.flatten()
print(f.shape)
plt.plot(f, 'x')


Q1=np.percentile(f, 25) ; # the value 25 is fixed for every problem;
Q3=np.percentile(f, 75) ; # the value 25 is fixed for every problem;
range=[Q1-1.5*(Q3-Q1),Q3+1.5*(Q3-Q1)];
pos = np.concatenate((np.where(f>range[1]),np.where(f<range[0])),axis=1)
newData = np.delete(rawTrainingData, pos, axis=0)

g = newData[:,[6]] 
g = g.flatten()
print(g.shape)
plt.xlim(0, 1000) # use the same axes setting as the above figure (with three outliers) to better reflect the difference
plt.ylim(0, 500)
plt.plot(g, 'x')
np.savetxt('ProcessedTrainingData.csv', newData, fmt='%.2f', delimiter=',', header=" T(t-2),T(t-1),T(t),D(t-2),D(t-1),D(t),P(t+1)")


fileName = 'Testing_Data.csv'
rawTestingData = pd.read_csv(fileName).values
f = rawTestingData[:,[6]] 
f = f.flatten()
plt.xlim(0, 520) # use the same axes setting as the above figure (with three outliers) to better reflect the difference
plt.ylim(0, 230)
plt.plot(f, 'x')
Q1=np.percentile(f, 25) ; # the value 25 is fixed for every problem;
Q3=np.percentile(f, 75) ; # the value 25 is fixed for every problem;
range=[Q1-1.5*(Q3-Q1),Q3+1.5*(Q3-Q1)];
pos = np.concatenate((np.where(f>range[1]),np.where(f<range[0])),axis=1)
newTestingData = np.delete(rawTestingData, pos, axis=0)
g = newTestingData[:,[6]] 
g = g.flatten()
print(g.shape)

plt.xlim(0, 520) # use the same axes setting as the above figure (with three outliers) to better reflect the difference
plt.ylim(0, 230)
plt.plot(g, 'x')

np.savetxt('ProcessedTestingData.csv', newTestingData, fmt='%.2f', delimiter=',', header=" T(t-2),T(t-1),T(t),D(t-2),D(t-1),D(t),P(t+1)")



a = newData[:,[0]] # T(t-2)
a = a.flatten()
b = newData[:,[1]] # T(t-1)
b = b.flatten()
c = newData[:,[2]] #T(t)
c = c.flatten()
d1 = newData[:,[3]] #D(t-2)
d1 = d1.flatten()
d2 = newData[:,[4]] #D(t-1)
d2 = d2.flatten()
d3 = newData[:,[5]] #D(t)
d3 = d3.flatten()
p = newData[:,[6]] # price
p = p.flatten()

v = np.array([a,b,c,d1,d2,d3,p])
CCM=np.corrcoef(v)
plt.matshow(CCM)
groups= ['t-2','t-1','t','d-2','d-1', 'd', 'p']
x_pos = np.arange(len(groups))
plt.xticks(x_pos,groups)
 
y_pos = np.arange(len(groups))
plt.yticks(y_pos,groups)
 
plt.show()

temperature = a


plt.hist(temperature)
plt.xlabel('T(t-2)');


print(np.min(a))
print(np.max(a))
temperature = ctrl.Antecedent(np.linspace(18,34,a.size), 'temperature')
print(temperature)

temperature['cold'] = fuzz.trimf(temperature.universe, [18, 18, 26])
temperature['warm'] = fuzz.trimf(temperature.universe, [20, 26, 34])
temperature['hot'] = fuzz.trimf(temperature.universe, [26, 34, 34])
temperature.view()

demand = d3


plt.hist(demand)
plt.xlabel('D(t)');


print(np.min(d3))
print(np.max(d3))
demand = ctrl.Antecedent(np.linspace(np.min(d3)-200,np.max(d3)+200,d3.size), 'demand')

demand['very little'] = fuzz.trimf(demand.universe, [3600, 3600, 4510])
demand['little'] = fuzz.trimf(demand.universe, [3600, 4500, 5250])
demand['middle'] = fuzz.trimf(demand.universe, [4500, 5250, 6000])
demand['high'] = fuzz.trimf(demand.universe, [5250, 6000, 6750])
demand['very high'] = fuzz.trimf(demand.universe, [6000, 6750, 6750])


demand.view()
prices = p
plt.hist(prices)
plt.xlabel('P(t+1)');
print(np.min(prices))
print(np.max(prices))
prices = ctrl.Antecedent(np.linspace(np.min(prices)-10,np.max(prices)+10,prices.size), 'price')
prices['low'] = fuzz.trimf(prices.universe, [0, 0, 30])
prices['medium'] = fuzz.trimf(prices.universe, [0, 30,62])
prices['high'] = fuzz.trimf(prices.universe, [30, 62, 62])
prices.view()


print(prices.universe.shape)
pric_low = fuzz.trimf(prices.universe, [0, 0, 30])


ravi = fuzz.interp_membership(prices.universe,pric_low, 15.17)

print("%.2f" % ravi)

#membersip for temperature
t_cold = fuzz.trimf(temperature.universe, [18, 18, 26])
t_warm = fuzz.trimf(temperature.universe, [20, 26, 34])
t_hot = fuzz.trimf(temperature.universe, [26, 34, 34])
#memebership for demand
d_verylittle = fuzz.trimf(demand.universe, [3600, 3600, 4510])
d_little = fuzz.trimf(demand.universe, [3600, 4500, 5250])
d_middle = fuzz.trimf(demand.universe, [4500, 5250, 6000])
d_high = fuzz.trimf(demand.universe, [5250, 6000, 6750])
d_veryhigh = fuzz.trimf(demand.universe, [6000, 6750, 6750])
#membersip for prices
p_low = fuzz.trimf(prices.universe, [0, 0, 30])
p_medium = fuzz.trimf(prices.universe, [0, 30,62])
p_high = fuzz.trimf(prices.universe, [30, 62, 62])

#evaluate rules
index = 0
m_t_cold = fuzz.interp_membership(temperature.universe,t_cold, a[0])
m_t_warm = fuzz.interp_membership(temperature.universe,t_warm, a[0])
m_t_hot = fuzz.interp_membership(temperature.universe,t_hot, a[0])
print(a[0])
print(m_t_cold)
print(m_t_warm)
print(m_t_hot)
for i in range(1,5):
    print(i)


