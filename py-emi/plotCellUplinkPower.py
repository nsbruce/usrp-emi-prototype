import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time as pytime
from scipy.signal import lfilter
import datetime
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages('Uplink_824MHz_849MHz.pdf')

d = np.fromfile('celluplink.dat', dtype=np.float64)

time = d[0::5]
power = d[1::5]
centroid = d[2::5]
occupancy = d[3::5]
threshold = d[4::5]

spectra = np.fromfile('celluplink_spectra.dat', dtype=np.float32)
spectra = np.reshape(spectra, (len(spectra)/25000, 25000))
#spectra = 10.*np.log10(spectra)

s = np.sum(spectra, axis=1)/(threshold*25000)
#print 10.*np.log10(s)

#plt.figure()	
#plt.plot(np.linspace(836.5-25, 836.5+25, 50000), spectra[-1])
t = time[-1] #pytime.time()
dt = datetime.datetime.utcfromtimestamp(time[0])

tptr = time[0]
bloop = []
increment = 10
window = 30*60
while tptr < t:
	bloop.append(len(time[(time >= tptr)&(time < (tptr + window))])/float(window))
	tptr += increment
print(bloop)
plt.figure()
plt.title('Fraction of 1s dumps with uplink activity (%d minute window)\n%s (UTC)' %(window/60., dt))
plt.xlabel('Days')
plt.ylabel('Fraction')
plt.plot((np.arange(len(bloop))/8640.), bloop)
plt.grid()
plt.tight_layout()
plt.draw()
#exit(0)
pp.savefig()


plt.figure()
plt.plot((time.astype(np.uint32)-time[0])/86400., 10.*np.log10(s), 'o')
#plt.stem(time.astype(np.uint32), np.zeros(len(time)))
#plt.ylim([0, 1.2])
plt.xlabel('Time (days)')
plt.ylabel('Estimated SNR (dB)')
plt.title('Transmission Events in 824-849 MHz Uplink\n%s (UTC)' %(dt))
plt.grid()
plt.xlim([0, 4])
plt.tight_layout()
plt.draw()
pp.savefig()

#exit(1)

#plt.figure()
#plt.plot(np.linspace(824, 849, 25000), 10.*np.log10(np.mean(spectra, axis=0))-10.*np.log10(np.mean(threshold)), label='Mean')
#plt.plot(np.linspace(824, 849, 25000), 10.*np.log10(np.max(spectra, axis=0))-10.*np.log10(np.mean(threshold)), label='Max')
#plt.grid()
#plt.xlim([824,849])
#plt.xlabel('Frequency (MHz)')
#plt.ylabel('dB (rel)')
#plt.tight_layout()
#plt.draw()



#plt.figure()
#for i in range(0, spectra.shape[0]):
#	where = np.where(spectra[i] > (power[i]))[0]
#	#plt.title('Occupancy = %.2f' %(occupancy[i]))
#	#plt.plot(centroid[i], power[i], 'ro')
#	plt.plot(np.linspace(824, 849, 50000), spectra[i])
#	plt.xlim([824, 849])
#	plt.plot([824, 849], [threshold[i]]*2)
#	#plt.plot(spectra[i])
#	#plt.plot(where, np.ones(len(where)), 'o')
#plt.show()
#
#
#exit(0)

#plt.figure()
#plt.hist(power-threshold, bins=31)
#plt.show()

#plt.figure()
#plt.title('Raw Events')
#plt.imshow(10.*np.log10(spectra), aspect='auto')
#plt.xlabel('Channel Idx')
#plt.ylabel('Event Idx')
#plt.tight_layout()
#plt.draw()
#pp.savefig()
#plt.figure()
#plt.plot(centroid, occupancy, 'o')
#plt.xlabel('Spectral Centroid (MHz)')
#plt.ylabel('Occupancy Fraction')
#plt.tight_layout()
#plt.draw()
#
#fig = plt.figure(figsize=(12,10))
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(centroid, power-threshold, occupancy)
#ax.set_xlabel('Spectral Centroid (MHz)')
#ax.set_ylabel('Excess Power (dB)')
#ax.set_zlabel('Spectral Occupancy (%%)')
#plt.show()

plt.show()
pp.close()