import sys 
from certifier.output.main import run

network_file = sys.argv[1]
batch_size = int(sys.argv[2])
eps = 0.01

run(network_file, batch_size, eps)