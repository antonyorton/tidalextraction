

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import exp1
import scipy.optimize as opt
import time
import os


def fit_params(filename,wellname,flow_rate = 1, start_stop = [0,100], dist_matrix = 'distances.csv', imagewell_dist_matrix = 'imagewell_distances.csv', fit_leaky = True,plot_results = True, results_to_csv = False):
    
    """Curve fitting of groundwater pumping test data to both the Theis W(u) function and the Walton W(u,R/L) function
    
        Uses Scipy.optimise.curve_fit
        
        filename: Name of csv file: first column is Time (column header 'Time'). The other columns have obs. well name
                 as their header and the groundwater response as the data (rows)
                 Supports any number of observation wells refering to the same time data from column 1.
        wellname: The name of the pumping/recharge test well
        flow_rate: Rate of puming/recharge into the test well
        start_stop: list/array with 0 entry = start of puming and 1 entry = end of pumping time.
        dist_matrix: Name of a csv file: the first column contains the names for all the observation wells. 
                     The remaining columns are named after (ie column header name) each test well. 
                     The data is the distance between the obs. well named in the first column and the test 
                     well named in the first row.
        imagewell_dist_matrix: Name of csv file: same data format as above for the distance matrix except 
                    that now the data is the distance from the observation well to the image well 
                    (ie for no flow bdry conditions). Column header must have the same name as the test well.
        fit_leaky: True/False, Set True to fit only a Theis curve
    
        RETURNS:
        1.resulttheis: Dataframe, Summary of Theis fitted parameters
        2.fittedtheis: Dataframe, Theis response curve for each column. Column 1 is time
        3.resultleak: Same as above for leaky curve fit
        4.fittedleaky: Same as above for leaky curve fit  
    
    """
   
    
    
    
    print("Time units (sec, min, hr or day) and distance units (m, cm and mm) are the responsibility of the user. Displayed results will be in those same units.")
    print("The user is responsible for supplying consistent units")
    
    starttest = start_stop[0]
    endtest = start_stop[1]
    Q = flow_rate
    
    #import data
    data=pd.read_csv(filename)
    #data=data.iloc[0:int(use_percentage_of_data*len(data))].copy() #slightly trimmed to not include recovery period

    #crop data prior to test start time
    t=data.values[:,0]
    startindex = np.argmax(t>=starttest)
    data=data.iloc[startindex::].copy()
    data.reset_index(inplace=True,drop=True)
    t=data.values[:,0]
    endindex = np.argmax(t>=endtest)

    ###################

    fittedtheis = pd.DataFrame(t,columns=['Time']) #DF to store theis curves
    fittedleaky = pd.DataFrame(t,columns=['Time'])

    #time since start of test
    t=t-t[0]
    
    #time for recovery
    trecov = t[::]-t[endindex]
    trecov[0:endindex]=0

    #radial distances
    distance_matrix = pd.read_csv(dist_matrix,index_col=0)
    if include_image_well==True:
        imwell_distance_matrix = pd.read_csv(imagewell_dist_matrix,index_col=0)


    #Optimise standard Theis
    allvals = data.values[:,1::]
    list1 = []

    for i in range(allvals.shape[-1]):

        #######################
        #optimisable function for standard Theis for T
        #function to be optimised (for T)
        R = distance_matrix.loc[list(data)[i+1]][wellname]
        if include_image_well:
            Rimage = imwell_distance_matrix.loc[list(data)[i+1]][wellname]
        else:
            Rimage = 0
            
        def func(t,T,S):
            ftheis = Q/4/np.pi/T*exp1(R**2*S/4/T/(t+1e-9))- Q/4/np.pi/T*exp1(R**2*S/4/T/(trecov+1e-9))
            if include_image_well:
                ftheis += Q/4/np.pi/T*exp1(Rimage**2*S/4/T/(t+1e-9)) - Q/4/np.pi/T*exp1(Rimage**2*S/4/T/(trecov+1e-9))		
            return ftheis
        ######################

        print(i)
        vals = allvals[:,i]
        try:
            popt, pcov = opt.curve_fit(func,t,vals,bounds=([0,1e-6],[1000,1e-1]))#,maxfev=40000)
        except RuntimeError:
            print('- Error: curve fit failed')
        list1.append([list(data)[i+1],np.sqrt(np.diag(pcov))[0]]+list(popt)+[R])
        fittedtheis[list(data)[i+1]]=func(t,*popt)  #save Theis curve to dataframe


    resulttheis = pd.DataFrame(list1,columns=['name','1std_dev','T','S','R_dist'])
    print(resulttheis)
    ##END Optimise standard Theis
                
    ###Leaky aquifer
    if fit_leaky:    
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
        allvals=allvals[0::]  #shorten the set (modify here if taking too long)
        tnew=t[0::]           #shorten the set (modify here if taking too long)
        list1 = []
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
            def funcwalton(t,T,S,L):
                fwalton = Q/4/np.pi/T*waltonwell(R**2*S/4/T/(t+1e-9),R/L) - Q/4/np.pi/T*waltonwell(R**2*S/4/T/(trecov+1e-9),R/L)
                if include_image_well:
                    fwalton += Q/4/np.pi/T*waltonwell(Rimage**2*S/4/T/(t+1e-9),Rimage/L) - Q/4/np.pi/T*waltonwell(Rimage**2*S/4/T/(trecov+1e-9),Rimage/L)
                return fwalton
            ##############################
            
            try:
                popt, pcov = opt.curve_fit(funcwalton,tnew,vals,bounds=([0,1e-6,10],[1000,1,10000]))
            except RuntimeError:
                print('- Error: curve fit failed')
            list1.append([list(data)[i+1],np.sqrt(np.diag(pcov))[0]]+list(popt)+[R])
            fittedleaky[list(data)[i+1]]=funcwalton(t,*popt)  #save Theis curve to dataframe
            print(i)
            
        resultleak = pd.DataFrame(list1,columns=['name','1std_dev','T','S','L','R_dist'])
        print(resultleak)
        print(time.time()-ttt)
        ##END Optimise leaky Walton well function


    if plot_results:
        #plotting - both
        allvals = data.values[:,1::]
        for i in range(len(resulttheis)):
            plt.plot(t,fittedtheis.values[:,i+1],'b',label='Theis W(u)')
            if fit_leaky:
                plt.plot(t,fittedleaky.values[:,i+1],'r',label='Leaky W(u,r/L)')
            plt.plot(t,allvals[:,i],'k')
            plt.xlabel('Time (in same units as input file)')
            plt.ylabel('Response (in same units as input file)')
            plt.title(wellname+' Recharge - obs well: '+resulttheis.iloc[i]['name'])
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.show()   
        
    if results_to_csv:   
        #Save data to file
        resulttheis.to_csv('output'+wellname+'_TheisParam.csv')
        fittedtheis.to_csv('output'+wellname+'_Theisfit.csv')  
        resultleak.to_csv('output'+wellname+'_LeakyParam.csv')
        fittedleaky.to_csv('output'+wellname+'_Leakyfit.csv')
    
    
    if fit_leaky: 
        return (resulttheis,fittedtheis,resultleak,fittedleaky)
    else:
        return (resulttheis,fittedtheis,[],[])
        
if __name__ == "__main__":
    
    dir1 = os.getcwd()
    os.chdir('C:\\Users\\A_Orton\\Desktop\\python_codes\\5_Tidalextraction\\groundwaterTestData\\recharge\\data')
    
    print('0. well 1')
    print('1. well 2')
    print('2. well 3')
    print('3. well 4')
    numwell=int(input('Choose well number (0,1,2,3):'))
    wnames = ['well 2','well 3','well 4','well 5']
    imwell = [True,True,False,False]
    flow_rate=[2.0,1.2,0.5,1.7]
    #test start and finish times in order of wells as specified in wnames
    startstop=[[2,9],[42,67],[12,24],[32,39]]

    #Well name
    wellname=wnames[numwell]
    filename = wnames[numwell]+'_smoothed.csv'
    include_image_well=imwell[numwell]
    starttest = startstop[numwell][0]
    endtest = startstop[numwell][1]
    
    Q = flow_rate[numwell]/1000*86400  #Flow rate in m3/day to be consistent with time input column of groundwater response data
    
    resulttheis,fittedtheis,resultleak,fittedleaky = fit_params(filename, wellname,flow_rate = Q, start_stop = [starttest,endtest],fit_leaky=True)
    
    os.chdir(dir1)
    
    