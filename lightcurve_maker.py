import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft
from scipy import interpolate
from symfit import parameters, variables, sin, cos, Fit
plt.close('all')

file = open('DataStuff.txt', 'r')
jday=2458560

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


fig, ax = plt.subplots(2)
plt.subplots_adjust(hspace=0.5)
ax[0].plot(time, Mag, '.')
ax[0].set_title("Before Airmass Correction")
ax[1].plot(time,airCorrMag,'.')
ax[1].set_title("After Airmass Correction")
plt.xlabel("Julian Date")
plt.ylabel("Differential Magnitude")
plt.show()

plt.savefig("Lightcurve.pdf")
plt.close()



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



def fourier_series(x, f, n=0):
    """
    Returns a symbolic fourier series of order `n`.

    :param n: Order of the fourier series.
    :param x: Independent variable
    :param f: Frequency of the fourier series
    """
    # Make the parameter objects for all the terms
    a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
    sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
    # Construct the series
    series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
                     for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    return series

x, y = variables('x, y')
w, = parameters('w')
model_dict = {y: fourier_series(x, f=((2*np.pi)/(time[n2max]-time[nmax])), n=10)}
print(model_dict)

fit = Fit(model_dict, x=time, y=airCorrMag)
fit_result = fit.execute()
print(fit_result)
plt.plot(time, fit.model(x=time, **fit_result.params).y, 'b')
plt.title("Fourier Model")
plt.xlabel("Julian Date")
plt.ylabel("Differential Magnitude")
plt.show()
plt.savefig("fourier_model.pdf")
plt.close()

xdata = np.array(time)
ydata = np.array(fit.model(x=time, **fit_result.params).y)

f = interpolate.interp1d(time, ydata)
tnew = np.linspace(time[0],time[-1],np.size(time))
dt = tnew[1]-tnew[0]
magnew = f(tnew)

plt.plot(time, ydata, '.')
plt.plot(tnew, magnew, '-')
plt.show()
plt.close()

'''
fourier = fft.fft(magnew)

n = tnew.size
freq = fft.fftfreq(n, d=1*dt)

plt.plot(freq, fourier, '.')
plt.ylim(-10, 40)
plt.xlim(-100, 100)
plt.show()

'''
