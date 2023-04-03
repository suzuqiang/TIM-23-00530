import os,sys
global Configure
global LIST
t = os.getcwd()
LIST = '\\' if t[:2] == 'C:' else '/'
t = 'TIM-23-00530'
T = sys.argv[0][:sys.argv[0].find(t)+len(t)]
#%%
def RetrievalPackage():
    FilePath = T + LIST
    Key = True
    PythonPathFile = FilePath+'Save'+LIST+'PythonPath.txt'
    PythonPath = [s for s in sys.path if ('python' in s and 'zip' in s) ][0]
    if not (os.path.exists(PythonPathFile)):
        with open(PythonPathFile,'w') as f:
            f.write(str(PythonPath))
        f.close()
        Key = False
    else:
        with open(PythonPathFile, "r") as f:
            LoadPythonPath = f.read()
        f.close()
        if not PythonPath[:-6] == LoadPythonPath[:-6]:
            Key = False
            PythonPath = [s for s in sys.path if ('python' in s and 'zip' in s) ][0]
            with open(PythonPathFile,'w') as f:
                f.write(str(PythonPath))
            f.close()
    if not Key:
        libs = {"torch","warnings","math","matplotlib","random","sklearn","argparse",
                "datetime","scipy","numpy","seaborn","pywt",
                }
        for lib in libs:
                os.system("pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple "+lib)
#%%
def SetSeed():
    seed_value = 1
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
#%%
def Parse():
    FilePath = T + LIST
    parser = argparse.ArgumentParser(description='Schrobine')
    parser.add_argument('--Epoch', default=50, type=int)
    parser.add_argument('--Batch', default=128, type=int)
    parser.add_argument('--WC', default=1,type=int, help="Working condition")
    parser.add_argument('--AN', default=True,type=bool, help="Noise addition")
    parser.add_argument('--SNR', default=10,type=int)
    parser.add_argument('--RN', default=0, type=int, help="Random Number")
    parser.add_argument('--NCV', default=5, type=int, help="Number of cross validations")
    parser.add_argument('--NSC', default=8, type=int, help="Number of sample categories")
    parser.add_argument('--NSPT', default=800, type=int, help="Number of samples per type")
    parser.add_argument('--LR', default=0.0025, type=float, help="Learning rate")
    parser.add_argument('--SL', default=4096, type=int, help="Sample length")
    parser.add_argument('--PS', default=True, type=bool, help="Print style")
    parser.add_argument('--FD', default=FilePath, type=str, help="File directory")
    parser.add_argument('--ASS', default=7, type=int, help="Accuracy sampling step")
    args = parser.parse_args()
    return args
RetrievalPackage()
import torch,warnings,math,matplotlib,random,sklearn,argparse,datetime,scipy.io,pywt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.pyplot import MultipleLocator
warnings.filterwarnings('ignore')
SetSeed()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Configure = Parse()
del T,t
