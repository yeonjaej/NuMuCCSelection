from ParseCAF import LoadData
from PlotTools import *
import matplotlib.pyplot as plt
import yaml
import glob
import os
plt.rcParams['text.usetex']=True
plt.rcParams['savefig.facecolor']='white'

def GetCFG():
    cfg="""
    PathVariables:
      InputPath: '/home/mueller/Projects/NuMuSelection/'
      InputFile: 'icarus_2500.flat.root'
      TreeName: 'recTree'
    GenSettings:
      BatchSize: 1000
      ReloadData: no
      Reset: yes
    """
    cfg = yaml.load(cfg, Loader=yaml.FullLoader)
    #cfg['PathVariables']['InputFile'] = 'gen-prodcorsika_genie_nooverburden_icarus_Oct2020_20201124T132115_recoSCEfix.flat.root'

    PlotCFGFile = open('cfg.yaml', 'r')
    PlotCFG = PlotCFGFile.read()
    PlotCFGFile.close()
    PlotCFG = yaml.load(PlotCFG, Loader=yaml.FullLoader)
    
    return cfg, PlotCFG

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
    cfg, PlotCFG = GetCFG()
    Prog = dict()
    Scores = {'NuScore': 0.4,
              'FlashMatchScore': 7.0}

    NuMuCCFullSelection = FullSelection(PlotCFG, Scores)
    if cfg['GenSettings']['ReloadData']:
        NuMuCCFullSelection.CSVLoad()
        NuMuCCFullSelection.DrawHists()
    else:
        Cosmics = GetFiles('cosmics')
        NuCosmics = GetFiles('nucosmics')
        for f in NuCosmics:
            Data, Prog = LoadData(f,
                                  cfg['GenSettings']['BatchSize'],
                                  Prog,
                                  Scores)
            NuMuCCFullSelection.ProcessData(Data)
            Prog = dict()
        NuMuCCFullSelection.CSVDump(Reset=cfg['GenSettings']['Reset'])
        #NuMuCCFullSelection.DrawHists()

if __name__ == '__main__':
    main()
