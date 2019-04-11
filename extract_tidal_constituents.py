
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt

""" Use extract_amplitudes_and_phase to extract amplitude and phase for a periodic frequency
    with desired frequencies to extract given (in cycles per hour) by constitfreq
    
    Can be used to predict tide """

num_constits = 3

data = import_tide_csv_qldopendata('BrisbaneBar_2016.csv')  #import tide data from csv file (specific format for this function)
constitfreq = import_tide_constituents()  #tidal constituent frequencies


def extract_amplitudes_and_phase(data, constitfreq, num_constits = 7):

    """LEAST SQUARES EXTRACTION OF TIDAL CONSTITUENT AMPLITUDE AND PHASE FOR GIVEN OBSERVATIONS
       data: 2D arraylike, column 1 is time (units of hour) and column 2 is water level 
       constitfreq: 1D arraylike of desired frequencies (in cycles per hour) ie the constituents
       num_constituents: the number of constituents to extract amplitude and phase data for 
       
       Returns: DataFrame  """
    
    if np.mod(len(data),2)==0:  #remove last observation if even number of observations supplied
        data=data[0:-1]

    T = data[:,0]   #water levels
    H = data[:,1]   #measurement times (units of hour)
    freqs = constitfreq[0:num_constits+1].copy()   #desired constituents (read from top of list down)

    num_N = int(len(H)/2 - 0.5)
    Nvals = np.arange(-num_N,num_N+1)  # a cycle (-N, -N+1, -N+2 .... N-1, N) for use later
    num_M = num_constits
    t0 = T[num_N]  #midpoint time observation of set
    deltaT = T[1]-T[0]


    A = np.zeros((2*num_N+1,2*num_M+1)) #big badboy matrix

    for i in range(2*num_N+1):      #construct matrix A
        lx = [np.cos(2*np.pi*freqs[j]*Nvals[i]*deltaT) for j in range(1,len(freqs))]
        ly = [np.sin(2*np.pi*freqs[j]*Nvals[i]*deltaT) for j in range(1,len(freqs))]
        A[i,:] = [1] + lx + ly

    # now we solve   (A^T*A)Z = (A^T)H   where Z = (x0,x1,...,xM,y1,y2,....,yM)

    ATH = np.dot(A.T,H)
    ATA = np.dot(A.T,A)
    Z = np.linalg.solve(ATA,ATH)  #the resulting Z = (x0,x1,...,xM,y1,y2,....,yM)
                                  #  where xj = Rjcos(phij - freqj*t0)
                                  #        yj = Rjsin(phij - freqj*t0)
                                  #  and Rj and phij are unknown (now known) amplitude and phase for each constituent


    X = Z[1:num_M+1]
    Y = Z[num_M+1::]

    amplitudes = [Z[0]]+[np.sqrt(X[i]**2+Y[i]**2) for i in range(num_M)]
    phases = [0]+[np.mod(np.arcsin(X[i]/amplitudes[i+1])+2*np.pi*freqs[i+1]*t0,2*np.pi) for i in range(num_M)]

    return pd.DataFrame(np.c_[amplitudes,freqs,phases],columns = ['amplitudes','freqs','phases'])


def import_tide_constituents(filename='tide_constituents.csv'):
    """ reads a csv file with the tidal constituents frequency (cycle per hour) in a column names frequency """
    constits = pd.read_csv(filename)
    constits = constits['frequency'].values
    constits = np.hstack((0,constits))  # add 0 as a constituent
    return constits


def import_tide_csv_qldopendata(filename):
    """Imports file and converts date to a pd.datetime column
       filename points to a csv file downloaded from https://www.msq.qld.gov.au/Tides/Open-data  
       
       returns a 2d array [time, water level]  with time in epoch hours   """
       
    tides = pd.read_csv(filename,sep='  ',header=None,engine='python')
    
    tides.columns=['input_datetime','swl']
    tides['input_datetime']=tides['input_datetime'].apply(lambda x: str(x))
    tides['input_datetime']=tides['input_datetime'].apply(lambda x: '0'+x if len(x)==11 else x)
    tides['date']=pd.to_datetime(tides['input_datetime'],format='%d%m%Y%H%M')
    tides = tides[['date','swl']] #extract relevant columns
    tides['date']=pd.to_numeric(tides['date'])  #convert from datetime to nanosecond
    tides['date']=tides['date']/3600e9  #convert from nanosecond to hours
    
    data = tides.values  #convert to array
    l1=[data[i] for i in range(len(data)) if np.abs(np.mod(data[i,0],1))<1e-9]  #extract only those readings on the hour
    data = np.array(l1)  #rename as array
    
    return data


def dft_to_remove_high_freq(filename = 'my_file.csv',max_freq_hz = 1/50/60/60, time_col='date', \
    level_col='swl',time_unit_of_input='day',use_manual_inverse=False, print_freqs=True):

    """NOTE: This needs updating as is very messy with various options for time unit unput"""
    
    #Import file and create new columns
    data = pd.read_csv(filename)
    data['time_sec']=data[time_col].copy()
    data['swl']=data[level_col].copy()

    samp_len=len(data)

    if time_unit == 'nanosecond':
        t=pd.to_numeric(data['time_sec']).values/1e9
    elif time_unit == 'day':
        t=pd.to_numeric(data['time_sec']).values*24*3600
    elif time_unit == 'hour':
        t=pd.to_numeric(data['time_sec']).values*3600
    elif time_unit == 'minute':
        t=pd.to_numeric(data['time_sec']).values*60
    elif time_unit == 'second':
        print('time unit supplied in seconds .. very good!')
    else:
        raise ValueError('time_unit not understood.. options are: nanosecond, day, hour, minute, or second')

    t=t-t[0]  #convert to start at t= 0
    f_s = 1/(t[1]-t[0]) #sampling rate per second

    x=data['swl'].values  #series data

    #Fourier transform
    X = fft.fft(x[0:samp_len])
    freqs = fft.fftfreq(samp_len)*f_s
    phases = np.angle(X)

    #Remove fast frequencies
    X[np.abs(freqs)>(max_freq_hz)]=0
    invtide=fft.ifft(X)
    
    plt.plot(t,x,'r')
    plt.plot(t,invtide.real,'k')
    plt.show()
    
    
    
    constit_data = pd.DataFrame(np.c_[np.abs(X)*2/samp_len,freqs*3600,phases],columns = ['Amp','Freq_cph','Phase'])
    constit_data.sort_values('Amp',inplace=True,ascending=False)
    constit_data=constit_data[constit_data['Freq_cph']>=0].copy()
    print(constit_data.head(50))
    
    return
    
    
    

#Test - recreate cosine waves
amplitudes = result['amplitudes'].values
phases = result['phases'].values
freqs=result['freqs'].values 
T = data[:,0]
H = data[:,1]   

#Test on input data
func = np.zeros(len(T)) + amplitudes[0]
for i in range(1,len(amplitudes)):
    func+=amplitudes[i]*np.cos(phases[i]-2*np.pi*freqs[i]*T-np.pi/2)

plt.plot(T,H,'r',T,func,'g')
plt.show()
plt.plot(T,H,'r',T,H-func,'g')
plt.show()
    
    
#predict on future year data
data2=import_tide_csv_qldopendata('BrisbaneBar_2017.csv')
Tnew = data2[:,0]
Hnew = data2[:,1]

func = np.zeros(len(Tnew)) + amplitudes[0]
for i in range(1,len(amplitudes)):
    func+=amplitudes[i]*np.cos(phases[i]-2*np.pi*freqs[i]*Tnew-np.pi/2)
    
plt.plot(Tnew,Hnew,'r',Tnew,func,'g')
plt.show()

