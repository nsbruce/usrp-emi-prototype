import struct
import zmq

class ProtocolError(Exception):
    pass

class CtrlPort:

    _CMD_REQ_SET_FREQ = 100
    _CMD_REQ_SET_GAIN = 101
    _CMD_REQ_SET_ANT  = 102

    _CMD_REP_OK  = 200
    _CMD_REP_BAD = 400

    def __init__(self, host, port=21233):

        self._endpoint = 'tcp://%s:%d' %(host, port)

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(self._endpoint)

    def setFreq(self, freq):

        # Form and send the request.
        req = struct.pack('hf', self._CMD_REQ_SET_FREQ, freq)
        self._socket.send(req)

        # Get the reply 
        rep  = self._socket.recv()
        code, = struct.unpack('h', rep)

        # Check the result
        if (code != self._CMD_REP_OK):
            raise ProtocolError('CMD_REP_BAD: Set frequency %f.' %(freq))

    def setAnt(self, ant):

        # Form and send the request.
        req = struct.pack('hI', self._CMD_REQ_SET_ANT, ant)
        self._socket.send(req)

        # Get the reply 
        rep  = self._socket.recv()
        code, = struct.unpack('h', rep)

        # Check the result
        if (code != self._CMD_REP_OK):
            raise ProtocolError('CMD_REP_BAD: Set antenna %d.' %(ant))

    def setGain(self, gain):

        # Form and send the request.
        req = struct.pack('hf', self._CMD_REQ_SET_GAIN, gain)
        self._socket.send(req)

        # Get the reply 
        rep  = self._socket.recv()
        code, = struct.unpack('h', rep)

        # Check the result
        if (code != self._CMD_REP_OK):
            raise ProtocolError('CMD_REP_BAD: Set gain %f.' %(gain))

