from ParseCAF import LoadData
from PlotTools import *
import matplotlib.pyplot as plt
import yaml
import glob
import os
plt.rcParams['text.usetex']=True
plt.rcParams['savefig.facecolor']='white'

def GetCFG():
    PlotCFGFile = open('cfg.yaml', 'r')
    PlotCFG = PlotCFGFile.read()
    PlotCFGFile.close()
    PlotCFG = yaml.load(PlotCFG, Loader=yaml.FullLoader)
    return PlotCFG

def GetFiles(Type, Reset=False):
    if Reset:
        Files = glob.glob(f'/icarus/data/users/mueller/NuMuSelection/{Type}/*flat.root')
        Files = [ x + '\n' for x in Files ]
        Out = open(f'Files_{Type}', 'w')
        Out.writelines(Files)
        Out.close()
    else:
        In = open(f'Files_{Type}', 'r')
        Files = In.readlines()
        In.close()
    
    return [ x.strip('\n') for x in Files ]

def main():
    PlotCFG = GetCFG()
    Prog = dict()
    Scores = {'NuScore': 0.4,
              'FlashMatchScore': 7.0}

    NuMuCCFullSelection = FullSelection(PlotCFG, Scores)
    if PlotCFG['GenSettings']['ReloadData']:
        NuMuCCFullSelection.CSVLoad()
        NuMuCCFullSelection.DrawHists()
    else:
        Cosmics = GetFiles('cosmics')
        NuCosmics = GetFiles('nucosmics')
        for i, f in enumerate(NuCosmics):
            Data, Prog = LoadData(f,
                                  PlotCFG['GenSettings']['BatchSize'],
                                  Prog,
                                  Scores)
            NuMuCCFullSelection.ProcessData(Data)
            Prog = dict()
            print(f'Completed file(s): {i+1}')
        NuMuCCFullSelection.CSVDump(Reset=PlotCFG['GenSettings']['Reset'])

if __name__ == '__main__':
    main()
