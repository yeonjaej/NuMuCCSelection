import numpy as np
import warnings
import uproot
import awkward as ak
from Selection import *
from CalcCAF import *
from Utilities import *

def AssociatedTree(Key):
    TreeNames = {
        "rec": 'rec.nspill',
        "rec.slc": 'rec.nslc',
        "rec.slc.reco.trk": 'rec.slc.reco.ntrk',
        "rec.crt_hits": 'rec.ncrt_hits',
        "rec.mc.nu": 'rec.mc.nnu',
        "rec.reco.trk": 'rec.reco.ntrk',
        "rec.reco.trk.chi2pid": 'rec.reco.ntrk',
        "rec.reco.trk.truth.matches": 'rec.reco.ntrk'}
    TreeMatch = ''
    for i, Tree in enumerate(TreeNames.keys()):
        if Key.startswith(Tree) and len(Tree) > len(TreeMatch): TreeMatch = Tree
    return TreeNames[TreeMatch]

def LoadData(File, sCount, Prog, Scores, POTInfo):
    Branches = [
        "rec.slc.vertex.x",
        "rec.slc.vertex.y",
        "rec.slc.vertex.z",
        "rec.slc.truth.index",
        "rec.slc.tmatch.pur",
        "rec.slc.tmatch.eff",
        "rec.slc.truth.iscc",
        "rec.slc.truth.pdg",
        "rec.slc.truth.E",
        "rec.slc.truth.position.x",
        "rec.slc.truth.position.y",
        "rec.slc.truth.position.z",
        "rec.slc.self",
        "rec.slc.nu_score",
        "rec.slc.fmatch.score",
        "rec.slc.fmatch.time",
        "rec.slc.fmatch_a.score",
        "rec.slc.is_clear_cosmic",
        "rec.slc.reco.trk.len",
        "rec.slc.reco.trk.costh",
        "rec.slc.reco.trk.phi",
        "rec.slc.reco.trk.start.x",
        "rec.slc.reco.trk.start.y",
        "rec.slc.reco.trk.start.z",
        "rec.slc.reco.trk.end.x",
        "rec.slc.reco.trk.end.y",
        "rec.slc.reco.trk.end.z",
        "rec.slc.reco.trk.ID",
        "rec.slc.reco.trk.slcID",
        "rec.slc.reco.trk.parent_is_primary",
        "rec.slc.reco.trk.chi2pid0.chi2_muon",
        "rec.slc.reco.trk.chi2pid0.chi2_proton",
        "rec.slc.reco.trk.chi2pid1.chi2_muon",
        "rec.slc.reco.trk.chi2pid1.chi2_proton",
        "rec.slc.reco.trk.chi2pid2.chi2_muon",
        "rec.slc.reco.trk.chi2pid2.chi2_proton",
        "rec.slc.reco.trk.bestplane",
        "rec.slc.reco.trk.mcsP.fwdP_muon",
        "rec.slc.reco.trk.rangeP.p_muon",
        "rec.slc.reco.trk.rangeP.p_proton",
        "rec.slc.reco.trk.crthit.hit.time",
        "rec.slc.reco.trk.crthit.distance",
        "rec.slc.reco.trk.crttrack.angle",
        "rec.slc.reco.trk.crttrack.time",
        "rec.slc.reco.trk.truth.bestmatch.G4ID",
        "rec.slc.reco.trk.truth.bestmatch.energy",
        "rec.slc.reco.trk.truth.p.pdg",
        "rec.slc.reco.trk.truth.p.planeVisE",
        "rec.slc.reco.trk.truth.p.gen.x",
        "rec.slc.reco.trk.truth.p.gen.y",
        "rec.slc.reco.trk.truth.p.gen.z",
        "rec.slc.reco.trk.truth.p.genp.x",
        "rec.slc.reco.trk.truth.p.genp.y",
        "rec.slc.reco.trk.truth.p.genp.z",
        "rec.slc.reco.trk.truth.p.end.x",
        "rec.slc.reco.trk.truth.p.end.y",
        "rec.slc.reco.trk.truth.p.end.z",
        "rec.slc.reco.trk.truth.p.length",
        "rec.slc.reco.trk.truth.p.contained",
        "rec.slc.reco.trk.truth.p.end_process",
        "rec.slc.reco.trk.truth.total_deposited_energy",
        "rec.slc.reco.trk.truth.p.start.x",
        "rec.slc.reco.trk.truth.p.start.y",
        "rec.slc.reco.trk.truth.p.start.z",
        "rec.slc.reco.trk.truth.p.startT",
        "rec.mc.nu.E",
        "rec.mc.nu.pdg",
        "rec.mc.nu.iscc",
        "rec.pass_flashtrig",
        "rec.crt_hits.time",
        "rec.hdr.pot",
        "rec.hdr.evt",
        "rec.hdr.fno",
        "rec.hdr.subrun",
        "rec.hdr.run",
        "rec.hdr.ngenevt",
        "rec.hdr.evt"]

    # First we open the input ROOT file using Uproot. We also want to initialize the Prog
    # dictionary if it is empty (first time opening the file). The Prog dictionary saves the
    # number of spills, slices, tracks, etc. that have been processed so far, which enables
    # proper sequential processing of the data.
    ROOTTree = uproot.open(File)
    Data = dict()
    if len(Prog.keys()) == 0:
        Prog= {'proc.rec.nspill': 0,
               'proc.rec.nslc': 0,
               'proc.rec.reco.ntrk': 0,
               'proc.rec.slc.reco.ntrk': 0,
               'proc.rec.ntrue_particles': 0,
               'proc.rec.ncrt_hits': 0,
               'proc.rec.mc.nnu': 0}


    # Load branches that are relevant for defining how many slices, tracks, etc. correspond
    # to the requested event count.
    Data['rec.nslc'] = ROOTTree['recTree'].array('rec.nslc',
                                                 entrystart=Prog['proc.rec.nspill'],
                                                 entrystop=Prog['proc.rec.nspill']+sCount)
    Data['rec.reco.ntrk'] = ROOTTree['recTree'].array('rec.reco.ntrk',
                                                          entrystart=Prog['proc.rec.nspill'],
                                                          entrystop=Prog['proc.rec.nspill']+sCount)
    Data['rec.slc.reco.ntrk'] = ROOTTree['recTree'].array('rec.slc.reco.ntrk',
                                                          entrystart=Prog['proc.rec.nslc'],
                                                          entrystop=Prog['proc.rec.nslc']+sCount)
    Data['rec.ncrt_hits'] = ROOTTree['recTree'].array('rec.ncrt_hits',
                                                      entrystart=Prog['proc.rec.nspill'],
                                                      entrystop=Prog['proc.rec.nspill']+sCount)
    Data['rec.mc.nnu'] = ROOTTree['recTree'].array('rec.mc.nnu',
                                                   entrystart=Prog['proc.rec.nspill'],
                                                   entrystop=Prog['proc.rec.nspill']+sCount)

    # Now we loop through each of the defined branches and load them into our data object. We
    # must take care to load up the corresponding number of slices, tracks, etc. for the requested
    # event count. This is accomplished through some helper functions.
    for b in Branches:
        Key = b
        MatchedTree = AssociatedTree(Key)
        s, e = MapStartEnd(Data[MatchedTree] if MatchedTree != 'rec.nspill' else list(),
                           sCount,
                           Prog['proc.'+MatchedTree])
        e = s + sCount
        d = ROOTTree['recTree'].array(Key, entrystart=s, entrystop=e)
        Data[b] = d.flatten()

    # This dictionary stores the total number of events, slices, tracks, etc. that have been read
    # from the input file. This is returned at the end and can be passed back into the function to
    # allow sequential processing of the data.
    Prog['proc.rec.nspill'] += sCount
    Prog['proc.rec.nslc'] += sum(Data['rec.nslc'])
    Prog['proc.rec.reco.ntrk'] += sum(Data['rec.reco.ntrk'])
    Prog['proc.rec.slc.reco.ntrk'] += sum(Data['rec.slc.reco.ntrk'].flatten())
    Prog['proc.rec.ncrt_hits'] += sum(Data['rec.ncrt_hits'])
    Prog['proc.rec.mc.nnu'] += sum(Data['rec.mc.nnu'])
    print(Prog)

    # It is natural to loop over slices, but this means that we need to group tracks (and other
    # sub-objects of slices) by their slice. We do this by grouping variable YYY according to
    # variable nYYY (stores the number of items per slice, e.g. tracks).
    Groupings = ["rec.slc.reco.trk"]
    for k in Data.keys():
        for g in Groupings:
            if k.startswith(g):
                Data[k] = Group(Data[k].flatten(), Data[g.replace(g.split(".")[-1], "n"+g.split(".")[-1])].flatten())

    # Some variables are stored once for some higher-level object where it may be convenient to have
    # access to it for each sub-object. We can broadcast such a variable over the sub-objects for
    # convenience (e.g. access to slice variable for each track).
    ToBroadcast = ["rec.pass_flashtrig"]
    BroadcastOver = "rec.nslc"
    for k in ToBroadcast:
        Data[k] = Broadcast(Data[k], Data[BroadcastOver])

    # Now we reach the stage where all "raw" variables are loaded. There are many quantities that we
    # need to calculate from these "raw" variables. For example, we will need to mark slices as
    # belonging to the fiducial volume, or define whether or not slices pass various selections. It
    # is convenient to define these in a separate file for organization purposes. The convention of
    # "raw" variables beginning with the "rec" tag and constructed quantities with the "con" tag,
    # while keeping all tags for the substructure (e.g. "slc") is convenient for organization.

    # Define tags for particle slice/track classification.
    DefineParticleTypes(Data)

    # Define volume cuts for slices/tracks.
    DefineVolumeCuts(Data)

    # Define various misc. quantities that will be useful later.
    DefineMiscQuantities(Data)

    # Define the primary match for cases where two slices match to the same neutrino.
    DefinePrimaries(Data)
    
    # Define the Pandora-based selection tags
    DefinePandoraCuts(Data)
    
    # Define the Flash Matching selection tags.
    DefineFlashMatchingCuts(Data)
    
    # Calorimetric PID variables.
    DefineCalorimetricPIDVariables(Data)

    # Primary track selection.
    DefinePrimaryTrack(Data)

    print('NGenEvent', NGenEvent(Data))
    print('NEvent', NEvent(Data))
    if len(POTInfo.keys()) == 0:
        POTInfo['NuPOT'] = NeutrinoPOT(Data)
        POTInfo['NuPOTScale'] = 6.6e20 / POTInfo['NuPOT']
        POTInfo['NuPerSpill'] =  (NGenEvent(Data) * 5e12) / POTInfo['NuPOT']
        print(f'Neutrino POT: {POTInfo["NuPOT"]}')
        print(f'Neutrino POT Scale: {POTInfo["NuPOTScale"]}')
        print(f'Spills per Neutrino: {1 / POTInfo["NuPerSpill"]}')
        print(f'Alt. Spills per Neutrino: {POTInfo["NuPOT"] / (NEvent(Data) * 5e12)}')
    else:
        POTInfo['CosPOT'] = NGenEvent(Data) * 5e12 / (1.0 - POTInfo['NuPerSpill'])
        POTInfo['CosPOTScale'] = 6.6e20 / POTInfo['CosPOT']
        POTInfo['CosPerSpill'] = (NGenEvent(Data) * 5e12) / POTInfo['CosPOT']
        print(f'Cosmic POT: {POTInfo["CosPOT"]}')
        print(f'Cosmic POT Scale: {POTInfo["CosPOTScale"]}')
        print(f'Spills per Cosmic: {1 / POTInfo["CosPerSpill"]}')
    
    return Data, Prog, POTInfo
