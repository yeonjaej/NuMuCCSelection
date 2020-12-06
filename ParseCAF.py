import numpy as np
import warnings
import uproot
import pandas as pd
import awkward as ak
from Selection import *

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

def Group(Var, NGroup, debug=False):
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
    End = Prog + sum(Branch)
    return Start, End

def Dist(x0, y0, z0, x1, y1, z1):
    return np.sqrt((x1-x0)**2+(y1-y0)**2+(z1-z0)**2)

def LoadData(File, sCount, Prog, Scores):
    Branches = [
        "slc.vertex.x",
        "slc.vertex.y",
        "slc.vertex.z",
        "slc.truth.index",
        "slc.tmatch.pur",
        "slc.tmatch.eff",
        "slc.truth.iscc",
        "slc.truth.pdg",
        "slc.truth.E",
        "slc.truth.position.x",
        "slc.truth.position.y",
        "slc.truth.position.z",
        "slc.self",
        "slc.nu_score",
        "slc.fmatch.score",
        "slc.fmatch.time",
        "slc.fmatch_a.score",
        "slc.is_clear_cosmic",
        "slc.reco.trk.len",
        "slc.reco.trk.costh",
        "slc.reco.trk.phi",
        "slc.reco.trk.start.x",
        "slc.reco.trk.start.y",
        "slc.reco.trk.start.z",
        "slc.reco.trk.end.x",
        "slc.reco.trk.end.y",
        "slc.reco.trk.end.z",
        "slc.reco.trk.ID",
        "slc.reco.trk.slcID",
        "slc.reco.ntrk",
        "slc.reco.trk.parent_is_primary",
        "slc.reco.trk.chi2pid0.chi2_muon",
        "slc.reco.trk.chi2pid0.chi2_proton",
        "slc.reco.trk.chi2pid1.chi2_muon",
        "slc.reco.trk.chi2pid1.chi2_proton",
        "slc.reco.trk.chi2pid2.chi2_muon",
        "slc.reco.trk.chi2pid2.chi2_proton",
        "slc.reco.trk.bestplane",
        "slc.reco.trk.mcsP.fwdP_muon",
        "slc.reco.trk.rangeP.p_muon",
        "slc.reco.trk.rangeP.p_proton",
        "slc.reco.trk.crthit.hit.time",
        "slc.reco.trk.crthit.distance",
        "slc.reco.trk.crttrack.angle",
        "slc.reco.trk.crttrack.time",
        "slc.reco.trk.truth.bestmatch.G4ID",
        "slc.reco.trk.truth.bestmatch.energy",
        "slc.reco.trk.truth.p.pdg",
        "slc.reco.trk.truth.p.planeVisE",
        "slc.reco.trk.truth.p.gen.x",
        "slc.reco.trk.truth.p.gen.y",
        "slc.reco.trk.truth.p.gen.z",
        "slc.reco.trk.truth.p.genp.x",
        "slc.reco.trk.truth.p.genp.y",
        "slc.reco.trk.truth.p.genp.z",
        "slc.reco.trk.truth.p.end.x",
        "slc.reco.trk.truth.p.end.y",
        "slc.reco.trk.truth.p.end.z",
        "slc.reco.trk.truth.p.length",
        "slc.reco.trk.truth.p.contained",
        "slc.reco.trk.truth.p.end_process",
        "slc.reco.trk.truth.total_deposited_energy",
        "mc.nu.E",
        "mc.nu.pdg",
        "mc.nu.iscc",
        "true_particles.G4ID",
        "true_particles.pdg",
        "pass_flashtrig",
        "crt_hits.time"]
    
    TreeNames = {
        "rec": 'nspill',
        "rec.slc": 'nslc',
        "rec.slc.reco.trk": 'reco.ntrk',
        "rec.true_particles": 'ntrue_particles',
        "rec.crt_hits": 'ncrt_hits',
        "rec.mc.nu": 'mc.nnu',
        "rec.reco.trk": 'reco.ntrk',
        "rec.reco.trk.chi2pid": 'reco.ntrk',
        "rec.reco.trk.truth.matches": 'reco.ntrk'}
    
    ROOTTree = uproot.open(File)
    Data = dict()
    if len(Prog.keys()) == 0:
        Prog= {'proc.nspill': 0,
               'proc.nslc': 0,
               'proc.reco.ntrk': 0,
               'proc.ntrue_particles': 0,
               'proc.ncrt_hits': 0,
               'proc.mc.nnu': 0}

    Data['nslc'] = ROOTTree['recTree']['rec'].array('rec.nslc', entrystart=Prog['proc.nspill'], entrystop=Prog['proc.nspill']+sCount)
    Data['reco.ntrk'] = ROOTTree['recTree']['rec'].array('rec.reco.ntrk', entrystart=Prog['proc.nspill'], entrystop=Prog['proc.nspill']+sCount)
    Data['ntrue_particles'] = ROOTTree['recTree']['rec'].array('rec.ntrue_particles', entrystart=Prog['proc.nspill'], entrystop=Prog['proc.nspill']+sCount)
    Data['ncrt_hits'] = ROOTTree['recTree']['rec'].array('rec.ncrt_hits', entrystart=Prog['proc.nspill'], entrystop=Prog['proc.nspill']+sCount)
    Data['mc.nnu'] = ROOTTree['recTree']['rec'].array('rec.mc.nnu', entrystart=Prog['proc.nspill'], entrystop=Prog['proc.nspill']+sCount)

    
    for b in Branches:
        Key = "rec." + b
        for t in TreeNames.keys():
            if not Key.startswith(t):
                continue
            try:
                s, e = MapStartEnd(Data[TreeNames[t]] if TreeNames[t] != 'nspill' else list(), sCount, Prog['proc.'+TreeNames[t]])
                d = ROOTTree['recTree'][t].array(Key, entrystart=s, entrystop=e)
            except KeyError:
                continue
            Data[b] = d
            break
        else:
            raise KeyError(Key)

    Prog['proc.nspill'] += sCount
    Prog['proc.nslc'] += sum(Data['nslc'])
    Prog['proc.reco.ntrk'] += sum(Data['reco.ntrk'])
    Prog['proc.ntrue_particles'] += sum(Data['ntrue_particles'])
    Prog['proc.ncrt_hits'] += sum(Data['ncrt_hits'])
    Prog['proc.mc.nnu'] += sum(Data['mc.nnu'])
    
    Groupings = ["slc.reco.trk"]
    for k in Data.keys():
        for g in Groupings:
            if k.startswith(g):
                Data[k] = Group(Data[k], Data[g.replace(g.split(".")[-1], "n"+g.split(".")[-1])])

    ToBroadcast = ["pass_flashtrig"]
    BroadcastOver = "nslc"
    for k in ToBroadcast:
        Data[k] = Broadcast(Data[k], Data[BroadcastOver])
        
    Data['con.is_nu'] = Data['slc.truth.index'] >= 0
    Data['con.is_mu'] = (np.abs(Data['slc.reco.trk.truth.p.pdg']) == 13)
    Data['con.is_cosmic'] = np.invert(Data['con.is_nu'])
    Data['con.is_numu_cc'] = Data['con.is_nu'] & Data['slc.truth.iscc'] & (np.abs(Data['slc.truth.pdg']) == 14)
    Data['con.is_nu_other'] = Data['con.is_nu'] & np.invert(Data['con.is_numu_cc'])
    Data['con.TruthInFV'] = InFV(Data['slc.truth.position.x'],
                                 Data['slc.truth.position.y'],
                                 Data['slc.truth.position.z'])
    Data['con.RecoInFV'] = InFV(Data['slc.vertex.x'],
                                 Data['slc.vertex.y'],
                                 Data['slc.vertex.z'])
    Data['con.mu_max_track'] = Data['slc.reco.trk.truth.p.length'][np.abs(Data['slc.reco.trk.truth.p.pdg']) == 13].max()
    
    # For the cases where two slices match to the same neutrino, figure out which match is "primary".
    # Use whichever slice gets more of the deposited energy.
    Data['con.match_is_primary'] = Data['slc.truth.iscc'] & False # Clones the column maintaining size.

    # Check the max match for each neutrino match.
    # Group each slice by spill

    #warnings.filterwarnings('error')

    IndexSpill = Group(Data['slc.truth.index'], Data['nslc'], debug=True)
    EffSpill = Group(Data['slc.tmatch.eff'], Data['nslc'])
    PrimarySpill = Group(Data['con.match_is_primary'], Data['nslc'])

    for i in range(len(IndexSpill)):
        IndexSpill[i][ IndexSpill[i] != -1 ] = i
        PrimarySpill = PrimarySpill | (EffSpill[ IndexSpill == i ].max() == EffSpill) & (EffSpill[ IndexSpill == i ].max() > 0.)
    Data['con.match_is_primary'] = PrimarySpill.flatten()

    # Set all cosmic matches to be primary
    Data['con.match_is_primary'] = Data['con.match_is_primary'] | Data['con.is_cosmic']

    Data['con.true_fv_numu_cc_primary'] = Data['con.TruthInFV'] & Data['con.is_numu_cc'] & Data['con.match_is_primary']
    Data['con.not_clear_cosmic'] = np.invert(Data['slc.is_clear_cosmic'])
    Data['con.true_fv_cosmic'] = Data['con.TruthInFV'] & Data['con.is_cosmic']
    Data['con.reco_fv_cosmic'] = Data['con.RecoInFV'] & Data['con.is_cosmic']

    NaNMask = np.invert(np.isnan(Data['slc.nu_score']))
    Data['con.nu_score'] = np.zeros(len(Data['slc.nu_score']), dtype=bool)
    Data['con.nu_score'][NaNMask] = Data['slc.nu_score'][NaNMask] > Scores['NuScore']

    Data['con.tpc_preselection'] = (Data['con.RecoInFV'] &
                                    Data['con.not_clear_cosmic'] &
                                    Data['con.nu_score'])
    
    Data['con.flash_score'] = np.zeros(len(Data['slc.fmatch.score']), dtype=bool)
    NaNMask = np.invert(np.isnan(Data['slc.fmatch.score']))
    Data['con.flash_score'][NaNMask] = Data['slc.fmatch.score'][NaNMask] < Scores['FlashMatchScore']
    #Data['con.true_fv_cosmic_nuscore'] = Data['con.true_fv_cosmic'] & Data['con.nu_score']
    #Data['con.true_fv_numu_cc_primary_nuscore']

    # Calorimetric PID variables.
    for chi2 in ['chi2_muon', 'chi2_proton']:
        Data['slc.reco.trk.bestplane.' + chi2] = Data['slc.reco.trk.chi2pid2.' + chi2]
        Data['slc.reco.trk.bestplane.' + chi2][Data['slc.reco.trk.bestplane'] == 0] = Data['slc.reco.trk.chi2pid0.' + chi2][Data['slc.reco.trk.bestplane'] == 0]
        Data['slc.reco.trk.bestplane.' + chi2][Data['slc.reco.trk.bestplane'] == 1] = Data['slc.reco.trk.chi2pid0.' + chi2][Data['slc.reco.trk.bestplane'] == 1]
    Data['con.is_muon'] = np.abs(Data['slc.reco.trk.truth.p.pdg']) == 13
    Data['con.is_proton'] = np.abs(Data['slc.reco.trk.truth.p.pdg']) == 2212
    Data['con.bestmatch_majority_energy'] = (Data['slc.reco.trk.truth.bestmatch.energy'] / Data['slc.reco.trk.truth.total_deposited_energy']) > 0.5
    Data['con.bestmatch_majority_planeVisE'] = (Data['slc.reco.trk.truth.bestmatch.energy'] / Data['slc.reco.trk.truth.p.planeVisE']) > 0.5
    Data['slc.reco.trk.truth.p.inAV'] = (InAV(Data['slc.reco.trk.truth.p.gen.x'],
                                             Data['slc.reco.trk.truth.p.gen.y'],
                                             Data['slc.reco.trk.truth.p.gen.z']) &
                                         InFV(Data['slc.reco.trk.truth.p.end.x'],
                                              Data['slc.reco.trk.truth.p.end.y'],
                                              Data['slc.reco.trk.truth.p.end.z'])) 
    Data['slc.reco.trk.contained'] = InAV(Data['slc.reco.trk.end.x'],
                                          Data['slc.reco.trk.end.y'],
                                          Data['slc.reco.trk.end.z'])
    Data['slc.reco.trk.not_contained'] = np.invert(Data['slc.reco.trk.contained'])
    Data['slc.reco.trk.truth.p.is_stopping'] = ((Data['slc.reco.trk.truth.p.end_process'] == 1) |
                                                (Data['slc.reco.trk.truth.p.end_process'] == 2) |
                                                (Data['slc.reco.trk.truth.p.end_process'] == 3) |
                                                (Data['slc.reco.trk.truth.p.end_process'] == 41))
    Data['slc.reco.trk.truth.p.not_stopping'] = np.invert(Data['slc.reco.trk.truth.p.is_stopping'])

    Data['slc.reco.trk.atslc'] = Dist(Data['slc.reco.trk.start.x'],
                                      Data['slc.reco.trk.start.y'],
                                      Data['slc.reco.trk.start.z'],
                                      Data['slc.vertex.x'],
                                      Data['slc.vertex.y'],
                                      Data['slc.vertex.z']) < 10.0

    # Primary track selection.
    PrimaryTrack = PrimaryTracks(Data)
    PrimaryTrackInd = ak.JaggedArray.fromcounts((PrimaryTrack >=0)*1, PrimaryTrack[PrimaryTrack >= 0])
    TruePrimaryTrack = TruePrimaryTracks(Data)
    TruePrimaryTrackInd = ak.JaggedArray.fromcounts((TruePrimaryTrack >=0)*1, TruePrimaryTrack[TruePrimaryTrack >= 0])

    Keys = list(Data.keys())
    for k in Keys:
        if k.startswith('slc.reco.trk'):
            SLCKey = k.replace('slc.reco.trk.', 'con.ptrk.')
            Data[SLCKey] = np.empty(Data['slc.nu_score'].shape)
            Data[SLCKey][:] = np.NaN
            IsBool = Data[k][0].dtype == 'bool'
            d = Data[k] + 0 # Copy
            Data[SLCKey][PrimaryTrack >= 0] = d[PrimaryTrackInd[PrimaryTrackInd >= 0]].flatten()
            if IsBool:
                Data[SLCKey] = Data[SLCKey] == 1 # Re-cast as bool if necessary.
    Data['con.has_ptrk'] = PrimaryTrack >= 0
    
    #HasTruePrimaryTack = TruePrimaryTrack >= 0
    #Valid = ((Data['slc.reco.trk.truth.p.length'] > 50.) &
    #         (Data['slc.reco.trk.truth.p.inAV'] | (Data['slc.reco.trk.truth.p.length'] > 100.0)) &
    #         Data['con.TruthInFV'])
    #Valid = ValidAllTypes & Data['con.is_numu_cc']
    #ValidPrimary = PrimaryTrack == np.nan
    #ValidPrimary[PrimaryTrackInd.count() > 0] = Valid[PrimaryTrackInd].flatten()
    #Data['con.trk.good_ptrack'] = PrimaryTrack[ValidPrimary & Data['con.match_is_primary']] == TruePrimaryTrack[ValidPrimary & Data['con.match_is_primary']]
    #Data['con.valid_matched_primary'] = ValidPrimary
    #Data['con.ptrack_and_pmatch'] = (ValidPrimary &
    #                                 Data['con.match_is_primary'])
    #Data['con.ptrack_and_pmatch_numu_cc'] = Data['con.ptrack_and_pmatch'] & Data['con.is_numu_cc']
    #Data['con.ptrack_and_pmatch'][Data['con.ptrack_and_pmatch']] = Data['con.trk.good_ptrack']
    #Data['con.ptrack_and_pmatch_numu_cc'][Data['con.ptrack_and_pmatch_numu_cc']] = Data['con.trk.good_ptrack']
    Data['con.ptrack_pion'] = np.abs(Data['con.ptrk.truth.p.pdg']) == 211#(ValidPrimary & (np.abs(Data['slc.reco.trk.truth.p.pdg']) == 211))
    Data['con.ptrack_proton'] = np.abs(Data['con.ptrk.truth.p.pdg']) == 2212#(ValidPrimary & (np.abs(Data['slc.reco.trk.truth.p.pdg']) == 2212))
    
    #for k in Data.keys():
    #    u, c = np.unique(np.isnan(Data[k]), return_counts=True)
    #    if True in u:
    #        i = np.argwhere(u)
    #        print(f'Key {k} contains {c[i]} NaN`s!')
    return Data, Prog
