import numpy as np
import awkward as ak

def Broadcast(Var, NBroadcast):
    return np.repeat(Var, NBroadcast)

def BroadcastAwk(Var, NBroadcast):
    FlatVar = Var.flatten()
    FlatNBroadcast = np.repeat(NBroadcast, Var.counts)
    FlatVarRepeat = np.repeat(FlatVar, FlatNBroadcast)

    Reindex = np.hstack([np.add.outer(np.arange(0, Var.counts[i]) * NBroadcast[i], np.arange(0, NBroadcast[i])).flatten("F") for i in range(len(NBroadcast))])
    Reindex += np.repeat(np.hstack([[0], np.cumsum(NBroadcast * Var.counts)[:-1]]), NBroadcast * Var.counts)
    FlatVarRepeatOrdered = FlatVarRepeat[Reindex]
    return Group(FlatVarRepeatOrdered, np.repeat(Var.counts, NBroadcast))

def Group(Var, NGroup):
    #print(f'Length of variable: {len(Var)}, jagged size: {len(NGroup)}')
    return ak.JaggedArray.fromcounts(NGroup, Var)

def PruneDictOnKey(Dict, Key):
    Result = dict()
    L = len(Dict[Key])
    for k in Dict.keys():
        if len(Dict[k]) == L:
            Result[k] = Dict[k]
    return pd.DataFrame(Result)

def MapStartEnd(Branch, Count, Prog):
    if len(Branch) == 0:
        return Prog, Prog+Count
    Start = Prog
    End = Prog + sum(Branch.flatten())
    return Start, End

def Dist(x0, y0, z0, x1, y1, z1):
    return np.sqrt((x1-x0)**2+(y1-y0)**2+(z1-z0)**2)
