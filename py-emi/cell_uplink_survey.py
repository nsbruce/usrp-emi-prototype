#!/usr/bin/env python
#
# RRL preliminary survey

import numpy as np
import matplotlib.pyplot as plt
import zmq
from CtrlPort import CtrlPort 
import sys
import os
import time

if __name__ == '__main__':

    USRP_Host = '192.168.100.85'
    data_port = 21234
    cp = CtrlPort(USRP_Host)

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect('tcp://%s:%s' %(USRP_Host, data_port))
    socket.setsockopt(zmq.SUBSCRIBE, '')

    cp.setGain(90)
    fc = (824+849)/2.
    cp.setFreq(fc*1e6)
    f = np.linspace(fc-25, fc+25, 50000)
    f = f[12500:37500]

    sigma2 = 10**1.7
    beta=0.01
    lam = 2 

    try:
        outfile0 = open('celluplink.dat', 'ab', 0)
        outfile1 = open('celluplink_spectra.dat', 'ab', 0)
        while True:
            message = socket.recv()
            ydata = np.fromstring(message, dtype=np.float32)
            ydata = np.fft.fftshift(ydata)
            ydata = ydata[12500:37500] # Trim to 25 MHz
            p = np.mean(ydata) 
            mean = 10.*np.log10(p)
            
            centroid = np.sum(ydata*f)/np.sum(ydata)
            
            occupancy = np.sum((10.*np.log10(ydata) > (mean - 3)).astype(np.int32))/25000.

            if np.mean(ydata) < lam*sigma2:
                sigma2 = beta*p + (1-beta)*sigma2
                print('%.2f dB (%.2f dB)' %(10.*np.log10(sigma2), mean))
            else:
                print(time.time(), mean, centroid, occupancy, sigma2)
                os.system('spd-say "event"')
                sys.stdout.flush()
                outfile0.write(np.asarray([time.time(), mean, centroid, occupancy, sigma2]).astype(np.float64))
                outfile1.write(ydata.astype(np.float32))
            #outfile1.write(ydata[6250:].tostring())

    except KeyboardInterrupt:
        print('Exiting.')
        outfile0.close()
        outfile1.close()
        socket.close()
        context.term()
        exit(0)
