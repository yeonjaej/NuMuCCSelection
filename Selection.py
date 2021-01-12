import pandas as pd
import numpy as np
import warnings
from Utilities import *

def InBeam(t):
    return (t > 1.600) & (t < 3.400) # Wiggle room, accounting for the fact that we need to re-center the beam window.

def InBeamTrue(t):
    return (t > 0) & (t < 1.596) # No wiggle room.

def InFV(x, y, z):
    #warnings.filterwarnings('ignore')
    xmin = -369.33 + 25
    xmax = -71.1 + 25
    ymin = -181.7 + 25
    ymax = 134.8 - 25
    zmin = -895.95 + 30
    zmax = 895.95 - 50
    
    NotAnyNaN = np.invert(np.isnan(x) | np.isnan(y) | np.isnan(z))
    Mask = NotAnyNaN | True
    #print(np.shape(NotAnyNaN))
    #print(sum(np.invert(NotAnyNaN)))
    #print(sum(NotAnyNaN))
    #Mask[AnyNaN]
    Mask[NotAnyNaN] = (x[NotAnyNaN] > xmin) & (x[NotAnyNaN] < xmax) & (y[NotAnyNaN] > ymin) & (y[NotAnyNaN] < ymax) & (z[NotAnyNaN] > zmin) & (z[NotAnyNaN] < zmax)
    #Mask = (x > xmin) & (x < xmax) & (y > ymin) & (y < ymax) & (z > zmin) & (z < zmax)
    return Mask

def InAV(x, y, z):
    warnings.filterwarnings('ignore')
    xmin = -369.33 + 5
    xmax = -71.1 + 5
    ymin = -181.7 + 5
    ymax = 134.8 - 5
    zmin = -895.95 + 5
    zmax = 895.95 - 5
    
    #NotAnyNaN = np.invert(np.isnan(x) | np.isnan(y) | np.isnan(z))
    #Mask = NotAnyNaN | True
    #Mask[NotAnyNaN]
    #Mask[NotAnyNaN] = (x[NotAnyNaN] > xmin) & (x[NotAnyNaN] < xmax) & (y[NotAnyNaN] > ymin) & (y[NotAnyNaN] < ymax) & (z[NotAnyNaN] > zmin) & (z[NotAnyNaN] < zmax)
    Mask = (x > xmin) & (x < xmax) & (y > ymin) & (y < ymax) & (z > zmin) & (z < zmax)
    return Mask

def NuScore(Scores):
    return Scores > 0.4

def FlashScore(Scores):
    return Scores < 7.0

def AtSLC(Data):
    x0 = Data['rec.slc.reco.trk.start.x']
    y0 = Data['rec.slc.reco.trk.start.y']
    z0 = Data['rec.slc.reco.trk.start.z']
    Ind = Data['rec.slc.reco.ntrk'].flatten().astype(np.intp)
    Shape = Data['rec.slc.reco.trk.start.x'].counts
    x1 = Group(np.repeat(np.array(Data['rec.slc.vertex.x'].flatten()), Ind), Shape)#.flatten()
    y1 = Group(np.repeat(np.array(Data['rec.slc.vertex.y'].flatten()), Ind), Shape)#.flatten()
    z1 = Group(np.repeat(np.array(Data['rec.slc.vertex.z'].flatten()), Ind), Shape)#.flatten()
    
    return (np.sqrt((x1-x0)**2+(y1-y0)**2+(z1-z0)**2) < 10.0)
    

def PrimaryTracks(Data):
    # Tracks considered as coming from the neutrino vertex.
    # We require that the start of the track is within 10 cm of the vertex of the slice and that the
    # parent of the track has been marked as primary. 
    FromSlice = Data['conr.trk.atslc'] & Data['rec.slc.reco.trk.parent_is_primary']

    # We select exiting muon candidates by requiring that the track is both not contained and is of
    # at least 100 cm in length.
    MaybeMuonExiting = np.invert(Data['conr.trk.contained']) & (Data['rec.slc.reco.trk.len'] > 100)

    # We select contained muon candidates by requiring that the track is contained, has appropriate
    # proton/muon chi^2 scores, and has a track length of at least 50 cm.
    MaybeMuonContained = (Data['conr.trk.contained'] &
                          (Data['conr.trk.bestplane.chi2_proton'] > 60) &
                          (Data['conr.trk.bestplane.chi2_muon'] < 30) &
                          (Data['rec.slc.reco.trk.len'] > 50))

    # Now we put this information together: we require that track is within 10 cm of the vertex of
    # the slice, the parent of the track has been marked as primary, and that the track has been
    # marked as either a exiting muon or contained muon candidate. Then we mark each slice based
    # upon whether or not it has at least one muon candidate.
    MaybeMuon = FromSlice & (MaybeMuonContained | MaybeMuonExiting)
    HasMaybeMuon = MaybeMuon.any()

    # Locate cases where all tracks are contained.
    # Essentially here we are selecting the maximum length track in each slice. We also check for
    # cases where there is no track in the slice or no muon candidate in the slice and setting
    # those track indices to -1.
    RetVal = (Data['rec.slc.reco.trk.len'] * MaybeMuon).argmax().max()
    RetVal[ RetVal < 0 ] = -1
    RetVal[ np.invert(HasMaybeMuon) ] = -1

    # RetVal is list of indices (of length #slices) which specifies which track in each slice is
    # the longest muon candidate. If there are none, the index is -1.
    return RetVal

def TruePrimaryTracks(Data):
    RetVal = ((np.abs(Data['rec.slc.reco.trk.truth.p.pdg']) == 13) &
              (Data['rec.slc.reco.trk.truth.bestmatch.energy'] / Data['rec.slc.reco.trk.truth.p.planeVisE'] > 0.5)).argmax().max()
    IsNuMuCC = ((Data['rec.slc.truth.index'] >= 0) &
                Data['rec.slc.truth.iscc'] &
                (np.abs(Data['rec.slc.truth.pdg']) == 14))
    RetVal[ np.invert(IsNuMuCC) ] = -1
    RetVal[ RetVal < 0 ] = -1
    return RetVal

def NeutrinoPOT(Data):
    _, Ind = np.unique(Data['rec.hdr.subrun'] + Data['rec.hdr.run']*100, return_index=True)
    return np.sum(Data['rec.hdr.pot'][Ind])

def NGenEvent(Data):
    _, Ind = np.unique(Data['rec.hdr.subrun'] + Data['rec.hdr.run']*100, return_index=True)
    return np.sum(Data['rec.hdr.ngenevt'][Ind])

def NEvent(Data):
    return len(Data['rec.hdr.evt'])

#def CosmicPOT(DataCosmic, DataNu):
    
