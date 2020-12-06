import pandas as pd
import numpy as np
import warnings

def InBeam(t):
    return (t > 1.600) & (t < 3.400) # Wiggle room, accounting for the fact that we need to re-center the beam window.

def TrueInBeam(t):
    return (t > 1.600) & (t < 3.238) # No wiggle room, again account for the need to re-center the beam window.

def InFV(x, y, z):
    warnings.filterwarnings('ignore')
    xmin = -369.33 + 25
    xmax = -71.1 + 25
    ymin = -181.7 + 25
    ymax = 134.8 - 25
    zmin = -895.95 + 30
    zmax = 895.95 - 50
    
    NotAnyNaN = np.invert(np.isnan(x) | np.isnan(y) | np.isnan(z))
    Mask = NotAnyNaN | True
    Mask[NotAnyNaN]
    Mask[NotAnyNaN] = (x[NotAnyNaN] > xmin) & (x[NotAnyNaN] < xmax) & (y[NotAnyNaN] > ymin) & (y[NotAnyNaN] < ymax) & (z[NotAnyNaN] > zmin) & (z[NotAnyNaN] < zmax)
    return Mask

def InAV(x, y, z):
    warnings.filterwarnings('ignore')
    xmin = -369.33 + 5
    xmax = -71.1 + 5
    ymin = -181.7 + 5
    ymax = 134.8 - 5
    zmin = -895.95 + 5
    zmax = 895.95 - 5
    
    NotAnyNaN = np.invert(np.isnan(x) | np.isnan(y) | np.isnan(z))
    Mask = NotAnyNaN | True
    Mask[NotAnyNaN]
    Mask[NotAnyNaN] = (x[NotAnyNaN] > xmin) & (x[NotAnyNaN] < xmax) & (y[NotAnyNaN] > ymin) & (y[NotAnyNaN] < ymax) & (z[NotAnyNaN] > zmin) & (z[NotAnyNaN] < zmax)
    return Mask

def PrimaryTracks(Data):
    # Tracks considered as coming from the neutrino vertex.
    # We require that the start of the track is within 10 cm of the vertex of the slice and that the
    # parent of the track has been marked as primary. 
    FromSlice = Data['slc.reco.trk.atslc'] & Data['slc.reco.trk.parent_is_primary']

    # We select exiting muon candidates by requiring that the track is both not contained and is of
    # at least 100 cm in length.
    MaybeMuonExiting = np.invert(Data['slc.reco.trk.contained']) & (Data['slc.reco.trk.len'] > 100)

    # We select contained muon candidates by requiring that the track is contained, has appropriate
    # proton/muon chi^2 scores, and has a track length of at least 50 cm.
    MaybeMuonContained = (Data['slc.reco.trk.contained'] &
                          (Data['slc.reco.trk.bestplane.chi2_proton'] > 60) &
                          (Data['slc.reco.trk.bestplane.chi2_muon'] < 30) &
                          (Data['slc.reco.trk.len'] > 50))

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
    RetVal = (Data['slc.reco.trk.len'] * MaybeMuon).argmax().max()
    RetVal[ RetVal < 0 ] = -1
    RetVal[ np.invert(HasMaybeMuon) ] = -1

    # RetVal is list of indices (of length #slices) which specifies which track in each slice is
    # the longest muon candidate. If there are none, the index is -1.
    return RetVal

def TruePrimaryTracks(Data):
    RetVal = ((np.abs(Data['slc.reco.trk.truth.p.pdg']) == 13) &
              (Data['slc.reco.trk.truth.bestmatch.energy'] / Data['slc.reco.trk.truth.p.planeVisE'] > 0.5)).argmax().max()
    IsNuMuCC = ((Data['slc.truth.index'] >= 0) &
                Data['slc.truth.iscc'] &
                (np.abs(Data['slc.truth.pdg']) == 14))
    RetVal[ np.invert(IsNuMuCC) ] = -1
    RetVal[ RetVal < 0 ] = -1
    return RetVal

def fdd():
    # Primary track selection.

    # First we retrieve the list of indices corresponding to the identified primary tracks. Then
    # we create PrimaryTrackInd, which is an JaggedArray (of length #slices) with a single entry
    # in each row containing the index of the longest muon candidate track (empty entry if no such
    # candidate.
    PrimaryTrack = PrimaryTracks(Data)
    PrimaryTrackInd = ak.JaggedArray.fromcounts((PrimaryTrack >=0)*1, PrimaryTrack[PrimaryTrack >= 0])

    # We do the same thing with the truth information.
    TruePrimaryTrack = TruePrimaryTracks(Data)
    TruePrimaryTrackInd = ak.JaggedArray.fromcounts((TruePrimaryTrack >=0)*1, TruePrimaryTrack[TruePrimaryTrack >= 0])
    HasTruePrimaryTack = TruePrimaryTrack >= 0

    
    Valid = ((Data['slc.reco.trk.truth.p.length'] > 50.) &
             (Data['slc.reco.trk.truth.p.inAV'] | (Data['slc.reco.trk.truth.p.length'] > 100.0)) &
             Data['con.is_numu_cc'] &
             Data['con.TruthInFV'])
    
    ValidPrimary = PrimaryTrack == np.nan
    ValidPrimary[PrimaryTrackInd.count() > 0] = Valid[PrimaryTrackInd].flatten()
    Data['con.trk.good_ptrack'] = PrimaryTrack[ValidPrimary & Data['con.match_is_primary']] == TruePrimaryTrack[ValidPrimary & Data['con.match_is_primary']]
    Data['con.valid_matched_primary'] = ValidPrimary
    Data['con.ptrack_and_pmatch'] = (ValidPrimary &
                                     Data['con.match_is_primary'])
    Data['con.ptrack_and_pmatch'][Data['con.ptrack_and_pmatch']] = Data['con.trk.good_ptrack']
    Data['con.ptrack_pion'] = (ValidPrimary & (np.abs(Data['slc.reco.trk.truth.p.pdg']) == 211))
    Data['con.ptrack_proton'] = (ValidPrimary & (np.abs(Data['slc.reco.trk.truth.p.pdg']) == 2212))
