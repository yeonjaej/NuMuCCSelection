import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

class FullSelection:
    def __init__(self, CFG, Scores):
        self.Flows = list()
        for f in CFG['Flows']:
            self.Flows.append(SelectionFlow(f))

        self.Cuts = list()
        for c in CFG['Cuts']:
            self.Cuts.append(Selection(c))

        self.Discriminations = list()
        for d in CFG['Discriminations']:
            self.Discriminations.append(Discrimination(d))
        
    def ProcessData(self, Data):
        for f in self.Flows: f.ProcessData(Data)
        for c in self.Cuts: c.ProcessData(Data)
        for d in self.Discriminations: d.ProcessData(Data)

    def CSVLoad(self):
        for f in self.Flows: f.CSVLoad()
        for c in self.Cuts: c.CSVLoad()
        for d in self.Discriminations: d.CSVLoad()

    def CSVDump(self, Reset=True):
        for f in self.Flows: f.CSVDump(Reset)
        for c in self.Cuts: c.CSVDump(Reset)
        for d in self.Discriminations: d.CSVDump(Reset)
        
    def DrawHists(self):
        for f in self.Flows: f.Draw()
        for c in self.Cuts: c.Draw()
        for d in self.Discriminations: d.Draw()
        
class Selection:
    def __init__(self, CFG):
        self.FigName = list(CFG.keys())[0]
        self.VarName = CFG[self.FigName]['Var']
        b = CFG[self.FigName]['Bins']
        self.Bins = np.linspace(b[0], b[1], b[2])
        self.BinContent = np.zeros((2, len(self.Bins)-1), dtype=int)
        self.Labels = CFG[self.FigName]['Labels']
        self.XLabel = CFG[self.FigName]['XLabel']
        self.Text = CFG[self.FigName]['Text']
        self.TextPos = CFG[self.FigName]['TextPos']
        self.Var = list()
        self.MaskVars = CFG[self.FigName]['MaskVar']
        self.Mask = list()
        self.SelName = CFG[self.FigName]['SelVar']
        self.SelMask = list()
        self.Vars = list()
        self.Fig = plt.figure(tight_layout=True)
        self.Ax = self.Fig.add_subplot(1,1,1)
        self.DisplayRemaining = CFG[self.FigName]['Remaining']
        
    def ProcessData(self, Data):
        self.Var = Data[self.VarName]
        self.Mask = np.ones(len(self.Var), dtype=bool)
        for m in self.MaskVars: self.Mask = self.Mask & Data[m]
        self.SelMask = Data[self.SelName]
        self.Vars = [self.Var[self.Mask & self.SelMask],
                     self.Var[self.Mask]]
        BinCounts = np.array((2,len(self.Bins)-1))
        for b in range(len(self.Vars)):
            self.BinContent[b] += np.histogram(self.Vars[b],
                                               bins=len(self.Bins)-1,
                                               range=(min(self.Bins), max(self.Bins)))[0]
    def Draw(self):
        for l in range(len(self.Labels)):
            self.Ax.hist(self.Bins[:-1],
                         histtype='step',
                         label=self.Labels[l],
                         bins=self.Bins,
                         weights=self.BinContent[l],
                         linewidth=3)
        self.Ax.set_xlabel(self.XLabel)
        self.Ax.set_ylabel('Entries')
        if self.DisplayRemaining:
            if self.Text != '': self.Text += f'\nRemaining: {round(100*(sum(self.BinContent[0])/sum(self.BinContent[1])),2)}\%'
            else: self.Text += f'Remaining: {round(100*(sum(self.BinContent[0])/sum(self.BinContent[1])),2)}\%'
        self.Ax.text(self.TextPos[0],
                     self.TextPos[1],
                     self.Text,
                     horizontalalignment='center',
                     fontsize=16,
                     transform=self.Ax.transAxes,
                     fontweight='extra bold')
        self.Ax.legend()
        self.Fig.savefig(f'Plots/{self.FigName}.png')
        
    def CSVLoad(self):
        DF = pd.read_csv(f'CSVs/{self.FigName}.csv')
        for i in range(len(self.Labels)): self.BinContent[i] = DF[f'c{i}'].to_numpy()
        
    def CSVDump(self, Reset=True):
        if Reset:
            Dict = {'b': range(len(self.Bins)-1)}
            for i in range(len(self.Vars)): Dict[f'c{i}'] = self.BinContent[i]
            DF = pd.DataFrame(Dict)
        else:
            DF = pd.read_csv(f'CSVs/{self.FigName}.csv')
            Dict = {'b': range(len(self.Bins)-1)}
            for i in range(len(self.Vars)): Dict[f'c{i}'] = self.BinContent[i]
            DF = DF.append(pd.DataFrame(Dict))
        DF.to_csv(f'CSVs/{self.FigName}.csv', index=False)

class Discrimination:
    def __init__(self, CFG):
        self.FigName = list(CFG.keys())[0]
        b = CFG[self.FigName]['Bins']
        self.Bins = np.linspace(b[0], b[1], b[2])
        self.VarName = CFG[self.FigName]['Var']
        self.XLabel = CFG[self.FigName]['XLabel']
        self.Text = CFG[self.FigName]['Text']
        self.TextPos = CFG[self.FigName]['TextPos']
        self.TypeMask = list()
        self.TypeLabels = list()
        self.TypeIsSignal = list()
        self.BinContent = list()
        self.PassFail = list()

        for t in CFG[self.FigName]['Types']:
            self.AddType(t['MaskVars'], t['Label'], t['Signal'])

        self.DoCut = False
        if CFG[self.FigName]['SelVar'] != '':
            self.SetSelection(CFG[self.FigName]['SelVar'],
                              CFG[self.FigName]['Score'],
                              CFG[self.FigName]['CutAbove'])
            
        self.Fig = plt.figure(tight_layout=True)
        self.Ax = self.Fig.add_subplot(1,1,1)

    def AddType(self, MaskVar, Label, Signal=True):
        if isinstance(MaskVar, str):
            self.TypeMask.append([MaskVar])
        elif isinstance(MaskVar, list):
            self.TypeMask.append(MaskVar)
        elif MaskVar == None:
            self.TypeMask.append([])
        self.TypeLabels.append(Label)
        self.TypeIsSignal.append(Signal)
        self.BinContent = np.zeros((len(self.TypeMask), len(self.Bins)-1))
        self.PassFail.append(np.zeros(2))
        
    def SetSelection(self, SelVar, CutVal, CutAbove=True):
        self.SelVar = SelVar
        self.CutVal = CutVal
        self.CutAbove = CutAbove
        self.DoCut = True

    def ProcessData(self, Data):
        Vars = list()
        for v in range(len(self.TypeMask)):
            Mask = np.ones(len(Data[self.VarName]), dtype=bool)
            for m in self.TypeMask[v]: Mask = Mask & Data[m]
            Vars.append(Data[self.VarName][Mask].flatten())
            if self.DoCut:
                self.PassFail[v][0] += sum(Mask & Data[self.SelVar])
                self.PassFail[v][1] += sum(Mask & np.invert(Data[self.SelVar]) & np.invert(np.isnan(Data[self.VarName])))
        BinCounts = np.array((len(Vars), len(self.Bins)-1))
        for b in range(len(Vars)):
            self.BinContent[b] += np.histogram(Vars[b],
                                               bins=len(self.Bins)-1,
                                               range=(min(self.Bins), max(self.Bins)))[0]
    def Draw(self):
        for l in range(len(self.TypeLabels)):
            if self.DoCut:
                P = 100*(self.PassFail[l][0] / sum(self.PassFail[l]))
                P = P if self.TypeIsSignal[l] else 100-P
                T = 'Sel. ' if self.TypeIsSignal[l] else 'Rej. '
                T += str(round(P,2)) + '\%' 
                self.TypeLabels[l] += f' ({T})'
            self.Ax.hist(self.Bins[:-1],
                         histtype='step',
                         label=self.TypeLabels[l],
                         bins=self.Bins,
                         weights=self.BinContent[l],
                         linewidth=3)
        self.Ax.set_xlabel(self.XLabel)
        self.Ax.set_ylabel('Entries')
        self.Ax.text(self.TextPos[0],
                     self.TextPos[1],
                     self.Text,
                     horizontalalignment='center',
                     fontsize=16,
                     transform=self.Ax.transAxes,
                     fontweight='extra bold')

        if self.DoCut:
            CutVal = (self.CutVal - self.Ax.get_xlim()[0]) / (self.Ax.get_xlim()[1] - self.Ax.get_xlim()[0])
            ArrowLow = max(0.0, CutVal - 0.3)
            ArrowHigh = min(1.0, CutVal + 0.3)
            TextLow = (CutVal+ArrowLow)/2.0
            TextHigh = max((CutVal+ArrowHigh) / 2.0, TextLow + 0.3)
            L = mpl.lines.Line2D((CutVal,CutVal), (0., 1.1), color="black", transform=self.Ax.transAxes,)
            L.set_clip_on(False)
            self.Ax.add_line(L)
            c = ['Red', 'Green']
            if self.CutAbove: c = c[::-1]
            self.Ax.annotate("", xytext=(ArrowLow, 1.05), xy=(CutVal, 1.05),
                             arrowprops=dict(arrowstyle="<-", color=c[0]), xycoords=self.Ax.transAxes,)
            self.Ax.text(TextLow, 1.1, 'Selected' if self.CutAbove else 'Rejected',
                         transform=self.Ax.transAxes, fontsize=12, color=c[0], horizontalalignment='center')
    
            self.Ax.annotate("", xytext=(ArrowHigh, 1.05), xy=(CutVal, 1.05),
                             arrowprops=dict(arrowstyle="<-", color=c[1]), xycoords=self.Ax.transAxes,)
            self.Fig.text(TextHigh, 1.1, 'Rejected' if self.CutAbove else 'Selected', 
                          transform=self.Ax.transAxes, fontsize=12, color=c[1], horizontalalignment='center')
        
        self.Ax.legend()
        self.Fig.savefig(f'Plots/{self.FigName}.png')

    def CSVLoad(self):
        DF = pd.read_csv(f'CSVs/{self.FigName}.csv')
        for i in range(len(self.TypeMask)): self.BinContent[i] = DF[f'c{i}'].to_numpy()
        if self.DoCut:
            for i in range(len(self.TypeMask)): self.PassFail[i][0] = DF[f'p{i}'].to_numpy().max()
            for i in range(len(self.TypeMask)): self.PassFail[i][1] = DF[f'f{i}'].to_numpy().max()
        
    def CSVDump(self, Reset=True):
        if Reset:
            Dict = {'b': range(len(self.Bins)-1)}
            for i in range(len(self.TypeMask)):
                Dict[f'c{i}'] = self.BinContent[i]
                Dict[f'p{i}'] = np.full(len(self.Bins)-1, self.PassFail[i][0])
                Dict[f'f{i}'] = np.full(len(self.Bins)-1, self.PassFail[i][1])
            DF = pd.DataFrame(Dict)
        else:
            DF = pd.read_csv(f'CSVs/{self.FigName}.csv')
            Dict = {'b': range(len(self.Bins)-1)}
            for i in range(len(self.TypeMask)):
                Dict[f'c{i}'] = self.BinContent[i]
                Dict[f'p{i}'] = np.full(len(self.Bins)-1, self.PassFail[i][0])
                Dict[f'f{i}'] = np.full(len(self.Bins)-1, self.PassFail[i][1])
            DF = DF.append(pd.DataFrame(Dict))
        DF.to_csv(f'CSVs/{self.FigName}.csv', index=False)
            
class SelectionFlow:
    def __init__(self, CFG):
        self.Surviving = {'Cosmic': [0],
                          'NuMuCC': [0],
                          'OtherNu': [0]}
        self.SelMasks = list()
        self.SelLabels = ['All Slices']
        self.FigName = list(CFG.keys())[0]
        self.Fig = plt.figure(tight_layout=True)
        self.Ax = self.Fig.add_subplot()
        self.Width = 0.3

        for s in CFG[self.FigName]['Selections']:
            self.AddSelection(s['SelVar'], s['Name'])

    def AddSelection(self, Masks, Label):
        if isinstance(Masks, str):
            self.SelMasks.append([Masks])
        elif isinstance(Masks, list):
            self.SelMasks.append(Masks)
        self.SelLabels.append(Label)
        for k in self.Surviving.keys(): self.Surviving[k].append(0)

    def ProcessData(self, Data):
        # First we count the slices before selections are applied.
        self.Surviving['Cosmic'][0] += sum( Data['con.is_cosmic'] )
        self.Surviving['NuMuCC'][0] += sum( Data['con.is_numu_cc'] )
        self.Surviving['OtherNu'][0] += sum( (Data['con.is_nu'] & np.invert(Data['con.is_numu_cc'])) )

        # Create empty mask before looping through selections.
        Mask = np.ones(len(Data['slc.nu_score']), dtype=bool)

        # Now iterate through selections and calculate surviving slices at each step.
        for i in range(len(self.SelMasks)):
            for s in self.SelMasks[i]: Mask = Mask & Data[s]
            self.Surviving['Cosmic'][i+1] += sum( Mask & Data['con.is_cosmic'] )
            self.Surviving['NuMuCC'][i+1] += sum( Mask & Data['con.is_numu_cc'] )
            self.Surviving['OtherNu'][i+1] += sum( Mask & Data['con.is_nu'] & np.invert(Data['con.is_numu_cc']) )
            
    def Draw(self):
        Y = np.arange(len(self.SelLabels))[::-1]
        B0 = self.Ax.barh(Y-self.Width, self.Surviving['NuMuCC'], self.Width, label=r'$\nu_\mu$ CC')
        B1 = self.Ax.barh(Y, self.Surviving['Cosmic'], self.Width, label=r'Cosmics')
        B2 = self.Ax.barh(Y+self.Width, self.Surviving['OtherNu'], self.Width, label=r'Other $\nu$')
        self.Ax.set_xlabel('Slices')
        self.Ax.set_title('Selection by Category')
        self.Ax.set_yticks(Y)
        self.Ax.set_yticklabels(self.SelLabels)
        self.Ax.set_xscale('log')
        self.Ax.legend()
        self.Fig.savefig(f'Plots/{self.FigName}.png', dpi=300) 

    def CSVLoad(self):
        Types = ['Cosmic', 'NuMuCC', 'OtherNu']
        DF = pd.read_csv(f'CSVs/{self.FigName}.csv')
        for t in Types: self.Surviving[t] = DF[t].to_numpy()
        
    def CSVDump(self, Reset=True):
        Types = ['Cosmic', 'NuMuCC', 'OtherNu']
        if Reset:
            Dict = dict()
            for t in Types: Dict[t] = self.Surviving[t]
            DF = pd.DataFrame(Dict)
        else:
            DF = pd.read_csv(f'CSVs/{self.FigName}.csv')
            Dict = dict()
            for t in Types: Dict[t] = self.Surviving[t]
            DF = DF.append(pd.DataFrame(Dict))
        DF.to_csv(f'CSVs/{self.FigName}.csv', index=False)
