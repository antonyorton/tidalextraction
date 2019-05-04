import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad, fixed_quad
from scipy.special import exp1
import scipy.optimize as opt
import time
import os
from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename


def fit_params(filename,wellname,flow_rate = 1, start_stop = [0,100], \
    coordsfile = 'coordinates.csv', imwell_coordsfile = 'imagewell_coordinates.csv', \
    include_image_well=False,fit_leaky = True,points_for_fit = 50,plot_results = True, results_to_csv = False):
    
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
   
    print("No assumptions about units is made by the program. The user is responsible for supplying consistent units")
    
    starttest = start_stop[0]
    endtest = start_stop[1]
    Q = flow_rate
    
    #import data
    data=pd.read_csv(filename)

    
    #crop data prior to test start time
    t=data.values[:,0]  
    startindex = np.argmax(t>=starttest-0.5*(t[1]-t[0]))
    data=data.iloc[startindex::].copy()
    data.reset_index(inplace=True,drop=True)
    t=data.values[:,0]
    endindex = np.argmax(t>=endtest-0.5*(t[1]-t[0]))

    ###################
      
    #time since start of test
    t=t-t[0]
    
    #time for recovery
    trecov = t[::]-t[endindex]
    trecov[0:endindex]=0

    #shorten the set creates tnew and trecovnew to be used in all leaky analysis from here on
    if points_for_fit<len(t):
        ind1 = reduced_indicies(len(t),max_n=points_for_fit,stop_pumping_index=endindex)  
        t=t[ind1]
        trecov=trecov[ind1]        
    else:
        ind1 = np.arange(len(t))
        
    fittedtheis = pd.DataFrame(t,columns=['Time']) #DF to store theis curves
        
    coords = pd.read_csv(coordsfile,index_col=0)
    coordwell = coords.loc[wellname][['east','north']].values
    coords['distance'] = np.sqrt((coords['east']-coordwell[0])**2+(coords['north']-coordwell[1])**2)

    if include_image_well:
        img_coords = pd.read_csv(imwell_coordsfile,index_col=0)
        coordwell = img_coords.loc[wellname][['east','north']].values
        coords['img_distance'] = np.sqrt((coords['east']-coordwell[0])**2+(coords['north']-coordwell[1])**2)

    #Optimise standard Theis
    allvals = data.values[:,1::][ind1]
    list1 = []

    
    
    for i in range(allvals.shape[-1]):

        #######################
        #optimisable function for standard Theis for T
        #function to be optimised (for T)
        R = coords.loc[list(data)[i+1]]['distance']
        if include_image_well:
            Rimage = coords.loc[list(data)[i+1]]['img_distance']
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
        allvals = data.values[:,1::][ind1]
        
        
        fittedleaky = pd.DataFrame(t,columns=['Time'])
        
        list1 = []
        for i in range(allvals.shape[-1]):
            vals = allvals[:,i]

            
            #get R#############
            R = coords.loc[list(data)[i+1]]['distance']
            if include_image_well:
                Rimage = coords.loc[list(data)[i+1]]['img_distance']
            else:
                Rimage = 0
            ####################

            ###NEEDS TO BE FIXED UP: trecov AND t should be passed to function, not just t
            #############################
            ##optimisable Walton leakage function for T,N,R,a=R/L  where T is transmissivity, N wells all at distance R from monitoring wells
            def funcwalton(t,T,S,L):
                fwalton = Q/4/np.pi/T*waltonwell(R**2*S/4/T/(t+1e-9),R/L) - Q/4/np.pi/T*waltonwell(R**2*S/4/T/(trecov+1e-9),R/L)
                if include_image_well:
                    fwalton += Q/4/np.pi/T*waltonwell(Rimage**2*S/4/T/(t+1e-9),Rimage/L) - Q/4/np.pi/T*waltonwell(Rimage**2*S/4/T/(trecov+1e-9),Rimage/L)
                return fwalton
            ##############################
            
            try:
                popt, pcov = opt.curve_fit(funcwalton,t,vals,bounds=([0,1e-5,10],[400,0.1,5000]))
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
        tfull = data.values[:,0]
        allvals = data.values[:,1::]
        for i in range(len(resulttheis)):
            #plt.plot(t,fittedtheis.values[:,i+1],'b',label='Theis W(u)')
            if fit_leaky:
                plt.plot(t+tfull[0],fittedleaky.values[:,i+1],'r',label='Leaky W(u,r/L)')
            plt.plot(tfull,allvals[:,i],'k')
            plt.plot(tfull[ind1],allvals[:,i][ind1],'ob',fillstyle='none')
            plt.xlabel('Time (in same units as input file)')
            plt.ylabel('Response (in same units as input file)')
            plt.title('Test well: '+wellname+', Obs well: '+resulttheis.iloc[i]['name']+', T = '+str(resultleak.iloc[i]['T'])[0:7]+', R = '+str(resultleak.iloc[i]['R_dist'])[0:6])
            plt.legend(loc='lower right')
            plt.grid(True)
            mngr = plt.get_current_fig_manager()
            mngr.window.wm_geometry(newGeometry='725x450+50+50')
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


def reduced_indicies(N,max_n=50,stop_pumping_index=100):
    
    """ provides a shortened number of sample points (exponentially spaced) for a puming/recovery test
        
        N: (Int) the current number of sample points
        nrequired: (Int) The desired number of sample points
        stop_pumping_index: (Int) The index at which pumping stops and recovery begins
    
        returns: index of sample points to use
    
    """

    n = stop_pumping_index
    indicies = []
    
    if n>=N:
        M = max_n
    else:
        M = int(max_n/2)
    
    
    #Indexes for pumping part of test
    R1 = M/np.log(n-1)
    set1=[]
    for i in range(M):
        set1.append(np.exp(i/R1))
    set1 = np.array(set1)
    set1-=1
    
    #Indexes for recovery part of test
    if M != max_n:
        R2 = M/np.log(N-n)
        set2=[]
        for i in range(M):
            set2.append(np.exp(i/R2))
        set2 = np.array(set2) 
        set2 = set2-1+n 
        indicies = np.hstack((set1,set2)).astype(int)
    else:
        indicies = set1.astype(int)
    
    if indicies[-1] != N-1:
        indicies[-1] = N-1   #make sure last point is included
    
    print('new index of length '+str(len(np.unique(indicies)))+' created.')
    return np.unique(indicies)


    
if __name__ == "__main__":
    

    from tkinter import filedialog
    from tkinter import *
    from pathlib import Path

    print(" ")
    print(".... ")
    print("AQUIFER PARAMETER CURVE FITTING PROGRAM")
    print("Time units (sec, min, hr or day) and distance units (m, cm and mm) are the responsibility of the user. Displayed results will be in those same units.")
    print("This includes flow rates, start and stop times, coordinates and distances.")
    print("Be a good engineer and use consistent units.")
    print(".... ")
    print(" ")
    print("The following data files must exist with names as shown:")
    print(" ")
    print("1. testdetails.csv:  columns - 'testwell' (ie pumping well name), 'Q' (flow rate), 'start' (time of pumping start), 'stop' (time of pumping stop), 'include_image' (True/False, whether to include image wells) ")
    print("2. wellcoordinates.csv:  columns - 'name', 'east', 'north'  (the coordinates of all obs. wells and pumping wells)")
    print("3. imagewellcoordinates.csv:  (optional) As for wellcordinates.csv, except provides the coordinates of the image wells only (one per pumping well allowed, name should be the same as the relevant pumping well)")
    print(" ")
    print("The user is asked to select a well responses file which must have the format: column 1. label (row1) = Time, data = time, columns 2,3,4,... labels = obs well names, data = water level in each of the obs wells for time specified in column 1")
    print(" ")
    
    #User input file location data
    mywellname = input('To continue, please input pumping/recharge well name:')
    root = Tk()
    filedatadirectory = Path(filedialog.askdirectory(title = "Select data directory where csv files are located"))
    myfilename = filedialog.askopenfilename(title = "select well responses data file",filetypes = (("csv files","*.csv"),))
    root.destroy()
    
    dir1 = os.getcwd()
    os.chdir(filedatadirectory)
    
    #Extract parameters
    testdetails = filedatadirectory / 'testdetails.csv'
    testdetails = pd.read_csv(testdetails,index_col=0)
    include_image_well = testdetails.loc[mywellname]['include_image']
    mycoordsfile = filedatadirectory / 'wellcoordinates.csv'
    if include_image_well:
        myimagecoordsfile = os.path.join(filedatadirectory,'wellcoordinates.csv')
    else:
        myimagecoordsfile = ''
    
    #Get data from testdetails.csv file
    start11 = testdetails.loc[mywellname]['start']
    end11 = testdetails.loc[mywellname]['stop']
    myflow_rate = testdetails.loc[mywellname]['Q']

    
    num_points = int(input("Input desired number of points to use for fitting (reduce number of points to speed up):"))
 
    resulttheis,fittedtheis,resultleak,fitleak = fit_params(myfilename, mywellname,points_for_fit=num_points,flow_rate = myflow_rate, start_stop = [start11,end11],\
    coordsfile = mycoordsfile, imwell_coordsfile = myimagecoordsfile, \
    include_image_well=include_image_well,fit_leaky=True,results_to_csv = True)

    print('Parameters and fitted curves saved to folder where data files are located.')
    os.chdir(dir1)
    a1 = input("press any key to close.")
    