from ParseCAF import LoadData
from PlotTools import *
import matplotlib.pyplot as plt
import yaml
import glob
import os
plt.rcParams['text.usetex']=True
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['figure.max_open_warning'] = 30

def GetCFG():
    PlotCFGFile = open('cfg.yaml', 'r')
    PlotCFG = PlotCFGFile.read()
    PlotCFGFile.close()
    PlotCFG = yaml.load(PlotCFG, Loader=yaml.FullLoader)
    return PlotCFG

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
        POTInfo = dict()
        Cosmics = '/home/mueller/Projects/NuMuSelection/icarus_cosmics.flat.root'
        NuCosmics = '/home/mueller/Projects/NuMuSelection/icarus_nucosmics.flat.root'

        Data, Prog, POTInfo = LoadData(NuCosmics,
                                       PlotCFG['GenSettings']['BatchSize'],
                                       Prog,
                                       Scores,
                                       POTInfo)
        NuMuCCFullSelection.ProcessData(Data, POTInfo['NuPOTScale'])
        Prog = dict()
        print('Completed nucosmics.')

        #Data, Prog, POTInfo = LoadData(Cosmics,
        #                               PlotCFG['GenSettings']['BatchSize'],
        #                               Prog,
        #                               Scores,
        #                               POTInfo)
        #NuMuCCFullSelection.ProcessData(Data, POTInfo['CosPOTScale'])
        #Prog = dict()
        #print('Completed cosmics.')
        
        NuMuCCFullSelection.CSVDump(Reset=PlotCFG['GenSettings']['Reset'])
        NuMuCCFullSelection.DrawHists()

if __name__ == '__main__':
    main()
