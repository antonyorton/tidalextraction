import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import exp1
import scipy.optimize as opt
import time

"""Code in progress which can be used to extract groundwater parameters via Theis curve or W(u,r/L) (leaky) curve fitting"""


#Well name
wellname='Dummy1'
include_image_well=False

#import data
data=pd.read_csv(wellname+'smoothed.csv')
data=data.iloc[0:int(0.5*len(data))].copy() #slightly trimmed to not include recovery period
t=data.values[:,0]
t=t-t[0]
t=t*86400

#known parameters
E = 80000  #elastic modulus  (2x SPT N of 40)
D = 5       #aquifer thickness
Q = 2.0       #flow rate L/scipy

#derived
S = 10/1.4/E*D  #Storativity
Q = Q/1000   #Q in m3/s

#radial distances
distance_matrix = pd.read_csv('distances.csv',index_col=0)
if include_image_well:
	imwell_distance_matrix = pd.read_csv('imagewell_distances.csv',index_col=0)





#dummy function for test data
#def testfunc(t,T,N,R):
#    return N*Q/4/np.pi/T*exp1(R**2*S/4/T/(t+1e-7))
#testfunc = np.vectorize(testfunc)

#v1 = testfunc(t,1e-3,1,200)
#v2 = testfunc(t,1e-4,1,450)
#v3 = testfunc(t,1e-5,1,100)
#v4 = testfunc(t,1e-6,1,150)
#data['v1']=v1
#data['v2']=v2
#data['v3']=v3
#data['v4']=v4   

#Optimise standard Theis
allvals = data.values[:,1::]
list1 = []
fittedtheis = pd.DataFrame(t,columns=['Time_s'])
for i in range(allvals.shape[-1]):

	#######################
	#optimisable function for standard Theis for T
	#function to be optimised (for T)
	R = distance_matrix.loc[list(data)[i+1]][wellname]
	if include_image_well:
		Rimage = imwell_distance_matrix.loc[list(data)[i+1]][wellname]
	else:
		Rimage = 0
		
	def func(t,T):
		ftheis = Q/4/np.pi/T*exp1(R**2*S/4/T/t)
		if include_image_well:
			ftheis += Q/4/np.pi/T*exp1(Rimage**2*S/4/T/t)	
		return ftheis
	######################


	vals = allvals[:,i]
	popt, pcov = opt.curve_fit(func,t,vals)
	list1.append([list(data)[i+1],np.sqrt(np.diag(pcov))[0]]+list(popt)+[R])
	fittedtheis[list(data)[i+1]]=func(t,popt)  #save Theis curve to dataframe
	
resulttheis = pd.DataFrame(list1,columns=['name','1std_dev','T_m2s','atR_dist'])
resulttheis['T_m2day']=86400*resulttheis['T_m2s']
##END Optimise standard Theis
       

       
       
    
###Leaky aquifer
    
## Walton leakage integrand for W(u,a) where a=r/L
def walt_integrand(y,a):
    return 1/y*np.exp(-y-1/4/y*a**2)
### Walton well (leaky) function W(u,a)
def waltonwell(u,a):
    return quad(walt_integrand,u,np.inf,args=(a))[0]
waltonwell = np.vectorize(waltonwell)    

ttt = time.time()
#Optimise leaky Walton well function
allvals = data.values[:,1::]
allvals=allvals[0::]  #shorten the set
tnew=t[0::]           #shorten the set
list1 = []
fittedleaky = pd.DataFrame(t,columns=['Time_s'])
for i in range(allvals.shape[-1]):
	vals = allvals[:,i]

	#get R#############
	R = distance_matrix.loc[list(data)[i+1]][wellname]
	if include_image_well:
		Rimage = imwell_distance_matrix.loc[list(data)[i+1]][wellname]
	else:
		Rimage = 0
	####################

	#############################
	##optimisable Walton leakage function for T,N,R,a=R/L  where T is transmissivity, N wells all at distance R from monitoring wells
	def funcwalton(t,T,L):
		fwalton = Q/4/np.pi/T*waltonwell(R**2*S/4/T/t,R/L)
		if include_image_well:
			fwalton += Q/4/np.pi/T*waltonwell(Rimage**2*S/4/T/t,Rimage/L)	
		return fwalton
	##############################

	popt, pcov = opt.curve_fit(funcwalton,tnew,vals,bounds=([0,10],[1,5000]))
	list1.append([list(data)[i+1],np.sqrt(np.diag(pcov))[0]]+list(popt)+[R])
	fittedleaky[list(data)[i+1]]=funcwalton(t,*popt)  #save leaky curve to dataframe
	print(i)
	
resultleak = pd.DataFrame(list1,columns=['name','1std_dev','T_m2s','L','atR_dist'])
resultleak['T_m2day']=86400*resultleak['T_m2s']
print(time.time()-ttt)
##END Optimise leaky Walton well function




#plotting - both
allvals = data.values[:,1::]
for i in range(len(resulttheis)):
	#popt1 = resulttheis.iloc[i]['T_m2s']
	plt.plot(t,fittedtheis.values[:,i+1],'b',label='Theis W(u)')
	#popt2 = resultleak.iloc[i][['T_m2s','L',]].values
	plt.plot(t,fittedleaky.values[:,i+1],'r',label='Leaky W(u,r/L)')
	plt.plot(t,allvals[:,i],'k')
	plt.title(wellname+ Recharge - obs well: '+resulttheis.iloc[i]['name'])
	plt.legend(loc='lower right')
	plt.grid(True)
	plt.show()   
    
    
    
