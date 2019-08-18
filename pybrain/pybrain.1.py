from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
from pybrain.structure import FullConnection

rede = FeedForwardNetwork()

inLayer = LinearLayer(2)
hiddenLayer = SigmoidLayer(3)
outLayer = LinearLayer(1)
bias = BiasUnit()
bias2 = BiasUnit()

rede.addModule(inLayer)
rede.addModule(hiddenLayer)
rede.addModule(outLayer)
rede.addModule(bias)
rede.addModule(bias2)

entradaoculta = FullConnection(inLayer, hiddenLayer)
ocultasaida = FullConnection(hiddenLayer, outLayer)
biasoculta = FullConnection(bias, hiddenLayer)
bias2oculta = FullConnection(bias2, outLayer)

rede.sortModules()

print(rede)
