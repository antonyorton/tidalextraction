import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.signal as sig
import matplotlib.pyplot as plt
import numpy.fft as fft


def optimise_tides7freq(t,vals,print_result=False):
    
	"""t in hours, 5 frequencies extracted
	
		returns (tidefunction, popt)
			which may be called by valsnew = tidefuction(tnew,*popt)"""
	
	#name, period, freq (cycles/hr)
	##   (1) m2	12.42,  0.080511
	##   (2) k1	23.93,  0.041781
	##   (3) o1 25.82,  0.038731
	##   (4) s2  12,    0.083333
	##   (5) n2	12.66,  0.078999
	##   (6) p1 14.959, 0.041553
	##   (7) q1	13.3987,0.037219
	

	def tidefunction(t,A0,A1,A2,A3,A4,A5,A6,A7,P1,P2,P3,P4,P5,P6,P7):
		
		q1,q2,q3,q4,q5,q6,q7 = 0.080511, 0.041781, 0.038731, 0.083333, 0.078999, 0.041553, 0.037219
		
		f = A0 +\
			A1*np.cos(2*np.pi*(q1*t - P1)) +\
			A2*np.cos(2*np.pi*(q2*t - P2)) +\
			A3*np.cos(2*np.pi*(q3*t - P3)) +\
			A4*np.cos(2*np.pi*(q4*t - P4)) +\
			A5*np.cos(2*np.pi*(q5*t - P5)) +\
			A6*np.cos(2*np.pi*(q6*t - P6)) +\
			A7*np.cos(2*np.pi*(q7*t - P7))
			
		return f

	popt,pcov = opt.curve_fit(tidefunction,t,vals,p0=(2,0.5,0.5,0.25,0.1,0.1,0.1,0.1,1.57,1.57,1.57,1.57,1.57,1.57,1.57),\
				bounds=([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[3,2,2,2,2,2,2,2,6.4,6.4,6.4,6.4,6.4,6.4,6.4]))

	#qvals for display only
	q1,q2,q3,q4,q5,q6,q7 = 0.080511, 0.041781, 0.038731, 0.083333, 0.078999, 0.041553, 0.037219
	qvals = np.array([0,q1,q2,q3,q4,q5,q6,q7])

	if print_result:
		data = pd.DataFrame(np.c_[qvals,popt[0:8],np.hstack((0,popt[8::]))],columns=['Frequency','Amplitude','Phase'])
		print(data)

	return (tidefunction,popt)

def optimise_tides5freq(t,vals,print_result=False):
    
	"""t in hours, 5 frequencies extracted
	
		returns (tidefunction, popt)
			which may be called by valsnew = tidefuction(tnew,*popt)"""
	
	#name, period, freq (cycles/hr)
	##   (1) m2	12.42,  0.080511
	##   (2) k1	23.93,  0.041781
	##   (3) o1 25.82,  0.038731
	##   (4) s2  12,    0.083333
	##   (5) n2	12.66,  0.078999
	##   (6) p1 14.959, 0.041553
	##   (7) q1	13.3987,0.037219
	

	def tidefunction(t,A0,A1,A2,A3,A4,A5,P1,P2,P3,P4,P5):
		
		q1,q2,q3,q4,q5 = 0.080511, 0.041781, 0.038731, 0.083333, 0.078999
		
		f = A0 +\
			A1*np.cos(2*np.pi*(q1*t - P1)) +\
			A2*np.cos(2*np.pi*(q2*t - P2)) +\
			A3*np.cos(2*np.pi*(q3*t - P3)) +\
			A4*np.cos(2*np.pi*(q4*t - P4)) +\
			A5*np.cos(2*np.pi*(q5*t - P5))
			
		return f

	popt,pcov = opt.curve_fit(tidefunction,t,vals,p0=(2,0.5,0.5,0.25,0.1,0.1,1.57,1.57,1.57,1.57,1.57),\
				bounds=([0,0,0,0,0,0,0,0,0,0,0],[3,2,2,2,2,2,6.4,6.4,6.4,6.4,6.4]))

	#qvals for display only
	q1,q2,q3,q4,q5 = 0.080511, 0.041781, 0.038731, 0.083333, 0.078999
	qvals = np.array([0,q1,q2,q3,q4,q5])

	if print_result:
		data = pd.DataFrame(np.c_[qvals,popt[0:6],np.hstack((0,popt[6::]))],columns=['Frequency','Amplitude','Phase'])
		print(data)

	return (tidefunction,popt)

def optimise_tides3freq(t,vals,print_result=False):
    
	"""t in hours, 3 frequencies extracted
	
		returns (tidefunction, popt)
			which may be called by valsnew = tidefuction(tnew,*popt)"""

	#name, period, freq (cycles/hr)
	##   (1) m2	12.42,  0.080511
	##   (2) k1	23.93,  0.041781
	#    (3) o1 25.82,  0.038731
	##   (4) s2  12,    0.083333
	##   (5) n2	12.66,  0.078999
	##   (6) p1 14.959, 0.041553
	##   (7) q1	13.3987,0.037219

	def tidefunction(t,A0,A1,A2,A3,P1,P2,P3):
		
		q1,q2,q3 = 0.080511, 0.041781, 0.038731
		
		f = A0 +\
			A1*np.cos(2*np.pi*(q1*t - P1)) +\
			A2*np.cos(2*np.pi*(q2*t - P2)) +\
			A3*np.cos(2*np.pi*(q3*t - P3))

			
		return f

	popt,pcov = opt.curve_fit(tidefunction,t,vals,p0=(1,0.5,0.25,0.1,1.57,1.57,1.57))#,\
				#bounds=([0,0,0,0,0,0,0],[np.inf,1.5,1,0.5,6.4,6.4,6.4]))

	#qvals for display only
	q1,q2,q3 = 0.080511, 0.041781, 0.038731
	qvals = np.array([0,q1,q2,q3])

	if print_result:
		data = pd.DataFrame(np.c_[qvals,popt[0:4],np.hstack((0,popt[4::]))],columns=['Frequency','Amplitude','Phase'])
		print(data)

	return (tidefunction,popt)
  
def optimise_tides3freq(t,vals,print_result=False):
    
	"""t in hours, 3 frequencies extracted
	
		returns (tidefunction, popt)
			which may be called by valsnew = tidefuction(tnew,*popt)"""

	#name, period, freq (cycles/hr)
	##   (1) m2	12.42,  0.080511
	##   (2) k1	23.93,  0.041781
	#    (3) o1 25.82,  0.038731
	##   (4) s2  12,    0.083333
	##   (5) n2	12.66,  0.078999
	##   (6) p1 14.959, 0.041553
	##   (7) q1	13.3987,0.037219

	def tidefunction(t,A0,A1,A2,P1,P2):
		
		q1,q2 = 0.080511, 0.041781
		
		f = A0 +\
			A1*np.cos(2*np.pi*(q1*t - P1)) +\
			A2*np.cos(2*np.pi*(q2*t - P2))

		return f

	popt,pcov = opt.curve_fit(tidefunction,t,vals,p0=(1,0.5,0.25,1.57,1.57))#,\
				#bounds=([0,0,0,0,0,0,0],[np.inf,1.5,1,0.5,6.4,6.4,6.4]))

	#qvals for display only
	q1,q2,q3 = 0.080511, 0.041781
	qvals = np.array([0,q1,q2])

	if print_result:
		data = pd.DataFrame(np.c_[qvals,popt[0:3],np.hstack((0,popt[3::]))],columns=['Frequency','Amplitude','Phase'])
		print(data)

	return (tidefunction,popt)

def fft_keeptidalbands(t,vals,bands = [[0.03,0.06],[0.07,0.1]]):

	"""Must pass t in hours
		Keeps frequencies that lie within tidal bands supplied
		"""
		
	t=t-t[0]  #convert to start at t= 0
	f_s = 1/(t[1]-t[0]) #sampling rate per second
	samp_len=len(t)
	x=vals[::]
	x=x-x[0]

	X = fft.fft(x[0:samp_len])
	freqs = fft.fftfreq(samp_len)*f_s
	phases = np.angle(X)

	#plt.plot(freqs,np.abs(X),'ob')
	#plt.grid('on')
	#plt.show()

	#Remove frequency (non tidal) bands		
	flag1_U = np.abs(freqs)<bands[0][0]
	
	flag2_L = np.abs(freqs)>bands[0][1]
	flag2_U = np.abs(freqs)<bands[1][0]
	
	flag3_L = np.abs(freqs)>bands[1][1]


	X[flag1_U]=0
	X[flag2_L&flag2_U]=0
	X[flag3_L]=0
	
	#plt.plot(freqs,np.abs(X),'ob')
	#plt.grid(True)
	#plt.show()
     
	invtide=fft.ifft(X)
	return invtide.real

def smooth_MM_tide(filename):
	
	""" 
	FUNCTION TO SMOOTH TIDALLY INFLUENCED GROUNDWATER
	filename points to a raw data excel csv file with:
	 - FIRST column: Excel date  (NOT minutes since start of test)
	 - OTHER columns: Water levels in wells at the times specified in the first column
					  with a different well on each column
	No data values will be filled with next valid value, or previous if there is no next value
	 - exclude: list noting the columns to be excluded from results
	
	OUTPUT: Saves a new csv file with the word 'smoothed' appended to the name
	"""

    ###Data preparation
	
	data = pd.read_csv(filename)
	data.dropna(axis=1,how='all',inplace=True)  #drop columns with all NaN
	data.fillna(method='bfill',inplace=True) #fill other NaN
	data.fillna(method='ffill',inplace=True)

	print("Excel date input DD-MM-YY HH:MM must be column 1 of input csv file")
	print("Ensure day is first, not month")
	###deal with excel date input (EXCEL DATE INPUT IS ASSUMED)
	###convert to numeric days


	col_1_name = list(data)[0]
	data[col_1_name] = pd.to_datetime(data[col_1_name],dayfirst=True)
	data[col_1_name] = pd.to_numeric(data[col_1_name])/1e9/24/3600  +25569.0    #converts to excel days


	#check if atmosphere 
	atminput = input('INPUT: Is atm data (converted to water height 1HPa = 0.01m (m)) included as last column of csv y/n?')
	if atminput == 'y':
		atmospheric_wlevel = data.values[:,-1]
		data = data.drop(labels = list(data)[-1],axis=1)

	t = data.values[:,0]*24 # put into hourly time (excel format is in days)

	


	#Main program is below:
	for i in range(1,len(list(data))):
		vals=data.values[:,i]
		vals=vals-vals[0]
		
		#DFT to keep roughly the specified 12hr, 24hr tidal bands only
		v2 = fft_keeptidalbands(t,vals,bands = [[0.03,0.06],[0.07,0.1]])
		t2 = t[::]
		
		#aim for the middle section of test data
		v2=v2[int(0*len(t)):int(1*len(t))]
		t2=t2[int(0*len(t)):int(1*len(t))]
	
		#Pull out specific tidal frequencies using scipy optimise
		func7,popt7 = optimise_tides7freq(t2,v2)
		smoothed_vals = vals-func7(t,*popt7)
	
		#ensure non negative and zero at t=0
		#smoothed_vals[smoothed_vals<0]=0
		#smoothed_vals[0]=0
		
		#plot - for information only
		plt.plot(t,vals,'r')
		plt.plot(t,smoothed_vals,'k')
		plt.plot(t,func7(t,*popt7),'g')
		plt.title(list(data)[i])
		plt.xlabel('Time (hr)')
		plt.ylabel('Water level (m)')
		plt.ylim((-0.2,1.6))
		plt.grid(True)

		if atminput == 'y':
			plt.plot(t,atmospheric_wlevel,'b', linewidth=0.2)

		plt.show()


		#replace in datafram
		data[list(data)[i]]=smoothed_vals

	#save to csv
	data.to_csv(filename[0:-4]+'smoothed.csv',index=False)
	
	return
	
	
	
	
if __name__ == "__main__":

    
	rtests = pd.read_csv('Dummy.csv')
	rtests['Time']=pd.to_datetime(rtests['Time'],dayfirst=True)
	rtests['Time']=pd.to_numeric(rtests['Time'])/1e9/3600
	names = list(rtests)
	t = rtests['Time'].values

	wellnnumber = input("Input well number (1,2,3 ..):")
	vals = rtests[names[int(wellnnumber)]].values
	
	#t2=t[50::]
	#v2=vals[50::]
	#v2=sig.detrend(v2)
	
	v2 = fft_keeptidalbands(t,vals)
	t2=t[::]
	
	#plt.plot(t,v2)
	#plt.show()
	
	v2=v2[20:int(0.8*len(t))]
	t2=t2[20:int(0.8*len(t))]
	
	#func3,popt3 = optimise_tides3freq(t2,v2)
	#plt.plot(t,vals-func3(t,*popt3),'r')
	func5,popt5 = optimise_tides5freq(t2,v2)
	plt.plot(t,vals-func5(t,*popt5),'g')
	plt.plot(t,vals,'b',linewidth=0.3)
	plt.show()