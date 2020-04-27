import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft
from scipy import interpolate
import pandas as pd

file = open('DataStuff.txt', 'r')
jday=248560

time=[]
flux=[]
err=[]
airMass=[]
fluxRef=[]
errRef=[]


for line in file:
	terms=line.split()
	time.append(float(terms[0]))
	flux.append(float(terms[1]))
	err.append(float(terms[2]))
	airMass.append(float(terms[3]))
	fluxRef.append(float(terms[4]))
	errRef.append(float(terms[5]))

tabletime = []
tableflux = []
tableerr = []
tableair = []
tablefluxr = []
tableerrr = []
for i in range(0, len(time)):
        if time[i] > 2458560.864 and time[i] < 2458560.888:
                tabletime.append(time[i])
                tableflux.append(flux[i])
                tableerr.append(err[i])
                tableair.append(airMass[i])
                tablefluxr.append(fluxRef[i])
                tableerrr.append(errRef[i])

Table = np.zeros((6, 19), order = 'F')
Table[0] = tabletime
Table[1] = tableflux
Table[2] = tableerr
Table[3] = tableair
Table[4] = tablefluxr
Table[5] = tableerrr

df = pd.DataFrame(Table)

## save to xlsx file

filepath = 'AstroTable.xlsx'

df.to_excel(filepath, index=False)

'''
time=np.array(time)
flux=np.array(flux)
err=np.array(err)
airMass=np.array(airMass)
fluxRef=np.array(fluxRef)
errRef=np.array(errRef)

RefAvg=np.average(fluxRef)
NormalizedRef=fluxRef/RefAvg

corrFlux=flux/NormalizedRef
corrMag=-2.5*np.log(corrFlux)/np.log(10)

Mag=-2.5*np.log(flux)/np.log(10)
k=0.1
airCorrMag=Mag+k*airMass

f = interpolate.interp1d(time, flux)
tnew = np.linspace(time[0],time[-1],np.size(time))
dt = tnew[1]-tnew[0]
fluxnew = f(tnew)




fig, ax = plt.subplots(2)
ax[0].plot(time, Mag, '.')
ax[0].set_title("Before Airmass Correction")
ax[1].plot(time,airCorrMag,'.')
ax[1].set_title("After Airmass Correction")
plt.xlabel("Julian Date")
plt.ylabel("Differential Magnitude")
plt.show()

maxi=0
nmax=0
for i in range(20,120):
	if airCorrMag[i]>maxi:
		maxi=airCorrMag[i]
		nmax=i
maxi2=0
n2max=0
for i in range(412,490):
	if airCorrMag[i]>maxi2:
		maxi2=airCorrMag[i]
		n2max=i

print((time[n2max]-time[nmax])*24)

'''
