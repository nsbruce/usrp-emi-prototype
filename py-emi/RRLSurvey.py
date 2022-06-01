#!/usr/bin/env python
#
# RRL preliminary survey

import numpy as np
import matplotlib.pyplot as plt
import zmq
from CtrlPort import CtrlPort 
import sys

if __name__ == '__main__':

    freqs = np.asarray([
        2383.4099,
        2362.3740,
        2340.8503,
        2318.8246,
        2296.2822,
        2273.2078,
        2249.5857,
        2225.3994,
        2200.6321,
        2175.2663,
        2149.2836,
        2122.6653,
        2095.3917,
        2067.4427,
        2038.7972,
        2009.4332,
        1979.3283,
        1948.4587,
        1916.8002,
        1884.3273,
        1851.0138,
        1816.8321,
        1783.1679, 
        1748.9862, 
        1715.6727, 
        1683.1998, 
        1651.5413, 
        1620.6717, 
        1590.5668, 
        1561.2028, 
        1532.5573, 
        1504.6083, 
        1477.3347, 
        1450.7164, 
        1424.7337, 
        1399.3679, 
        1374.6006, 
        1350.4143, 
        1326.7922, 
        1303.7178, 
        1281.1754, 
        1259.1497, 
        1237.6260, 
        1216.5901, 
        1196.0282, 
        1175.9271, 
        1156.2739, 
        1137.0563, 
        1118.2622, 
        1099.8800, 
        1081.8984, 
        1064.3068, 
        1047.0944, 
        1030.2512, 
        1013.7674, 
        997.6333, 
        981.8398, 
        966.3779, 
        951.2389, 
        936.4146, 
        921.8966, 
        907.6773,
        892.3227,
        878.1034,
        863.5854,
        848.7611,
        833.6221,
        818.1602,
        802.3667,
        786.2326,
        769.7488,
        752.9056,
        735.6932,
        718.1016,
        700.1200,
        681.7378,
        662.9437,
        643.7261,
        624.0729,
        603.9718
    ])[::-1]

    USRP_Host = '192.168.100.85'
    data_port = 21234
    cp = CtrlPort(USRP_Host)
    dead_time_sec = 2
    dwell_time_sec = 300

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect('tcp://%s:%s' %(USRP_Host, data_port))
    socket.setsockopt(zmq.SUBSCRIBE, '')

    cp.setGain(90) # TODO: is there one good static gain setting?
    loopcnt = 0


    try:
        while True:
            loopcnt += 1
            print('Starting Iteration %d' %(loopcnt))
            for f in freqs:
                print('Switching to frequency %.4f MHz' %(f))
                cp.setFreq(f*1e6)
                outfile0 = open('%.4f.avg.dat' %(f), 'ab')
                #outfile1 = open('%.4f.max.dat' %(f), 'ab')

                for i in np.arange(dead_time_sec):
                    message = socket.recv()

                for i in np.arange(dwell_time_sec):
                    message = socket.recv()
                    ydata = np.fromstring(message, dtype=np.float32)
                    print('.')
                    sys.stdout.flush()
                    outfile0.write(ydata.tostring())
                    #outfile1.write(ydata[6250:].tostring())
                print()

                outfile0.close()
                #outfile1.close()
    except KeyboardInterrupt:
        print('Exiting.')
        outfile0.close()
        socket.close()
        context.term()
        exit(0)
