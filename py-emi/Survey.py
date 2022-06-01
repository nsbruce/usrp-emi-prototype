#!/usr/bin/env python
#
# RRL preliminary survey

import numpy as np
import matplotlib.pyplot as plt
import zmq
from CtrlPort import CtrlPort 
import sys
import random
import os
import datetime

if __name__ == '__main__':

    freqs = np.arange(280, 1961, 80)

    basePath = '/opt/survey'
    surveyPath = os.path.join(basePath, datetime.datetime.utcnow().isoformat())

    os.mkdir(surveyPath)

    USRP_Host = 'localhost'
    data_port = 21234
    cp = CtrlPort(USRP_Host)
    dead_time_sec = 1
    dwell_time_sec = 1

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect('tcp://%s:%s' %(USRP_Host, data_port))
    socket.setsockopt(zmq.SUBSCRIBE, '')

    cp.setGain(70) # TODO: is there one good static gain setting?
    loopcnt = 0

    tsFile = open(os.path.join(surveyPath, 'ts.txt'), 'w')

    try:
        while True:
            loopcnt += 1
            print('Starting Iteration %d' %(loopcnt))
            random.shuffle(freqs)
            tsFile.write(datetime.datetime.utcnow().isoformat() +'\n')
            tsFile.flush()
            for f in freqs:
                print('Switching to frequency %.4f MHz' %(f))
                cp.setFreq(f*1e6)
                outfile0 = open('%s/%.4f.avg.dat' %(surveyPath, f), 'ab')
                #outfile1 = open('%.4f.max.dat' %(f), 'ab')

                for i in np.arange(dead_time_sec):
                    message = socket.recv()

                for i in np.arange(dwell_time_sec):
                    message = socket.recv()
                    ydata = np.fromstring(message, dtype=np.float32)
                    print '.',
                    sys.stdout.flush()
                    outfile0.write(ydata.tostring())
                    #outfile1.write(ydata[6250:].tostring())
                print ''

                outfile0.close()
                #outfile1.close()
    except KeyboardInterrupt:
        print('Exiting.')
        outfile0.close()
        socket.close()
        context.term()
        tsFile.close()
        exit(0)
