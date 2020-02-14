import os
import platform
import sys
from multiprocessing import cpu_count
from torch import cuda as check_gpu

# Set ROOT location of the git repository
try:
    if platform.node() == 'luca.home' or platform.node() == 'Luca.local' or platform.node() == 'Luca':
        ROOT_DIR = '/Users/luca/Projects/rl_sepsis'
    else: # Default to luca OSX
        ROOT_DIR = '/Users/luca/Projects/rl_sepsis'
        raise ValueError(('Conf.py WARNING: Unknown platform: '+platform.node()+". Defaulting to luca.home"))
    
except ValueError as err:
    print(err.args) 
    
# Add additional directories    
DATA_DIR = os.path.join(ROOT_DIR, "SEPSIS", "data")
EXP_DIR = os.path.join(ROOT_DIR, "SEPSIS", "experiments")

def print_python_environment():
    print("Current version of Python: " + str(platform.python_version()))
    print("Current environment of Python: " + str(sys.executable))
    print("Current working directory: " + str(os.getcwd()))
    print("Current platform: " + str(platform.node()))
    print("GPU available: ", check_gpu.is_available())
    print("cpu count: " , cpu_count())
