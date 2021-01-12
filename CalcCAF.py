import numpy as np
from Selection import *

def DefineParticleTypes(Data):
    # Creates boolean tags marking slices or tracks as belonging to various particle
    # classifications. None of these definitions are based on selection parameters.

    # Slices corresponding to neutrino events are distinguised by having a slice truth index
    # greater than 0.
    Data['cont.slc.is_nu'] = (Data['rec.slc.truth.index'] >= 0)
    
    # Mark muon tracks according to track particle truth PDG information.
    Data['cont.trk.is_muon'] = (np.abs(Data['rec.slc.reco.trk.truth.p.pdg']) == 13)

    # Slices which do not correspond to neutrino events are marked as cosmic slices.
    Data['cont.slc.is_cosmic'] = np.invert(Data['cont.slc.is_nu'])

    # Slices which both correspond to a neutrino event, are classified as a CC interaction, and
    # have the NuMu PDG code are marked as NuMu CC slices.
    Data['cont.slc.is_numu_cc'] = (Data['cont.slc.is_nu'] &
                                   Data['rec.slc.truth.iscc'] &
                                   (np.abs(Data['rec.slc.truth.pdg']) == 14))

    # Slices which correspond to a neutrino, but do not correspond to NuMu CC interactions are
    # marked as "other" nuetrinos.
    Data['cont.slc.is_nu_other'] = (Data['cont.slc.is_nu'] &
                                   np.invert(Data['cont.slc.is_numu_cc']))

    # Mark proton tracks according to track particle truth PDG information.
    Data['cont.trk.is_proton'] = (np.abs(Data['rec.slc.reco.trk.truth.p.pdg']) == 2212)

    # Mark (charged) pion tracks according to track particle truth PDG information.
    Data['cont.trk.is_pion'] = (np.abs(Data['rec.slc.reco.trk.truth.p.pdg']) == 211)

    # Mark tracks as in-time or out-of-time with the beam.
    Data['cont.trk.intime'] = InBeamTrue(Data['rec.slc.reco.trk.truth.p.startT'])
    Data['cont.trk.outtime'] = np.invert(Data['cont.trk.intime'])

    # Some boolean tags are being interpreted as int8's, which was a horribly frustrating problem
    # to debug. FORCE EVERYTHING.
    Data['cont.slc.is_nu'] = Data['cont.slc.is_nu'].astype(bool)
    Data['cont.trk.is_muon'] = Data['cont.trk.is_muon'].astype(bool)
    Data['cont.slc.is_cosmic'] = Data['cont.slc.is_cosmic'].astype(bool)
    Data['cont.slc.is_numu_cc'] = Data['cont.slc.is_numu_cc'].astype(bool)
    Data['cont.slc.is_nu_other'] = Data['cont.slc.is_nu_other'].astype(bool)
    Data['cont.trk.is_proton'] = Data['cont.trk.is_proton'].astype(bool)
    Data['cont.trk.is_pion'] = Data['cont.trk.is_pion'].astype(bool)
    Data['cont.trk.intime'] = Data['cont.trk.intime'].astype(bool)
    Data['cont.trk.outtime'] = Data['cont.trk.outtime'].astype(bool)
    
def DefineVolumeCuts(Data):
    # Creates boolean tags marking slices or tracks as belonging to either the fiducial or
    # active volume (as defined in Selection.py). Both truth and reco quantities are defined
    # here.

    # Mark slices as passing the fiducial volume cut (as defined in Selection.py) using both
    # truth and reco information.
    Data['cont.slc.TruthInFV'] = InFV(Data['rec.slc.truth.position.x'],
                                      Data['rec.slc.truth.position.y'],
                                      Data['rec.slc.truth.position.z'])
    
    Data['conr.slc.RecoInFV'] = InFV(Data['rec.slc.vertex.x'],
                                     Data['rec.slc.vertex.y'],
                                     Data['rec.slc.vertex.z'])
    
    # Mark tracks as passing the active volume cut (as defined in Selections.py) using truth
    # information. We require the track start in the AV and end in the FV.
    Data['cont.trk.TruthInAV'] = (InAV(Data['rec.slc.reco.trk.truth.p.gen.x'],
                                       Data['rec.slc.reco.trk.truth.p.gen.y'],
                                       Data['rec.slc.reco.trk.truth.p.gen.z']) &
                                  InFV(Data['rec.slc.reco.trk.truth.p.end.x'],
                                       Data['rec.slc.reco.trk.truth.p.end.y'],
                                       Data['rec.slc.reco.trk.truth.p.end.z']))

    # Mark tracks as passing the contained cut (defined as the track end being contained in the
    # fiducial volume according to Selection.py).
    Data['conr.trk.contained'] = InFV(Data['rec.slc.reco.trk.end.x'],
                                      Data['rec.slc.reco.trk.end.y'],
                                      Data['rec.slc.reco.trk.end.z'])

    # It is also convenient to define the inverse of the above quantity.
    Data['conr.trk.not_contained'] = np.invert(Data['conr.trk.contained'])

    # Since neutrinos were only generated in one cryostat (sigh...), we need to be able to
    # separate and ignore all cosmics outside of cryostat 0. Otherwise we inflate the number of
    # rejected cosmics.
    Data['cont.slc.cryo0'] = Data['rec.slc.vertex.x'] < 0

    # Some boolean tags are being interpreted as int8's, which was a horribly frustrating problem
    # to debug. FORCE EVERYTHING.
    Data['cont.slc.TruthInFV'] = Data['cont.slc.TruthInFV'].astype(bool)
    Data['conr.slc.RecoInFV'] = Data['conr.slc.RecoInFV'].astype(bool)
    Data['cont.trk.TruthInAV'] = Data['cont.trk.TruthInAV'].astype(bool)
    Data['conr.trk.contained'] = Data['conr.trk.contained'].astype(bool)
    Data['conr.trk.not_contained'] = Data['conr.trk.not_contained'].astype(bool)
    Data['cont.slc.cryo0'] = Data['cont.slc.cryo0'].astype(bool)
    
def DefineMiscQuantities(Data):
    # Defines various variables at both slice and track level using the "raw" information.

    # Define the maximum muon track length for each slice.
    Data['cont.slc.mu_max_track'] = Data['rec.slc.reco.trk.truth.p.length'][np.abs(Data['rec.slc.reco.trk.truth.p.pdg']) == 13].max()

    # Tag tracks as being either stopping or non-stopping tracks.
    Data['cont.trk.is_stopping'] = ((Data['rec.slc.reco.trk.truth.p.end_process'] == 1) |
                                    (Data['rec.slc.reco.trk.truth.p.end_process'] == 2) |
                                    (Data['rec.slc.reco.trk.truth.p.end_process'] == 3) |
                                    (Data['rec.slc.reco.trk.truth.p.end_process'] == 41))

    # And it is also useful to have the inverse of the previous tag.
    Data['cont.trk.not_stopping'] = np.invert(Data['cont.trk.is_stopping'])

    # Tag each track based on whether or not it starts sufficiently close to the slice verted (as
    # defined in Selection.py).
    Data['conr.trk.atslc'] = AtSLC(Data)

    # Some boolean tags are being interpreted as int8's, which was a horribly frustrating problem
    # to debug. FORCE EVERYTHING.
    Data['cont.trk.is_stopping'] = Data['cont.trk.is_stopping'].astype(bool)
    Data['cont.trk.not_stopping'] = Data['cont.trk.not_stopping'].astype(bool)
    Data['conr.trk.atslc'] = Data['conr.trk.atslc'].astype(bool)
    
def DefinePrimaries(Data):
    # There are cases where two slices match to the same neutrino. We need to figure out which
    # match is the "primary" match, which is defined as the matched slice with the most deposited
    # energy.

    # Clone the array while maintaining the proper length.
    Data['cont.slc.match_is_primary'] = Data['rec.slc.truth.iscc'] & False

    # We need to group the index, the efficiency, and the match_is_primary variable based on the
    # number of slices in each spill.
    IndexSpill = Group(Data['rec.slc.truth.index'], Data['rec.nslc'])
    EffSpill = Group(Data['rec.slc.tmatch.eff'], Data['rec.nslc'])
    PrimarySpill = Group(Data['cont.slc.match_is_primary'], Data['rec.nslc'])

    # Now we loop through the indices, select the slices which matched to the neutrino, and mark
    # the slice with the majority of the deposited energy as the primary match.
    
    for i in range(Data['rec.slc.truth.index'].max()+1):
        PrimarySpill = PrimarySpill | (EffSpill[IndexSpill==i].max() == EffSpill) & (EffSpill[IndexSpill==i].max() > 0)

    # Assign resulting boolean mask from flattened PrimarySpill.
    Data['cont.slc.match_is_primary'] = PrimarySpill.flatten()

    # All cosmics are considered to be primaries.
    Data['cont.slc.match_is_primary'] = (Data['cont.slc.match_is_primary'] | Data['cont.slc.is_cosmic'])

    # Some boolean tags are being interpreted as int8's, which was a horribly frustrating problem
    # to debug. FORCE EVERYTHING.
    Data['cont.slc.match_is_primary'] = Data['cont.slc.match_is_primary'].astype(bool)
    
def DefinePandoraCuts(Data):
    # Create boolean tags for each slice to mark them as passing several Pandora-related cuts.

    # Invert the Pandora clear cosmic boolean to indicate, when True, that the slice has not been
    # marked by Pandora to be a clear cosmic. This is done out of convenience.
    Data['conr.slc.not_clear_cosmic'] = np.invert(Data['rec.slc.is_clear_cosmic'].astype(bool))

    # We need to mark slices as passing/failing the Pandora nu-score cut, as defined in
    # Selection.py
    Data['conr.slc.nu_score'] = np.zeros(len(Data['rec.slc.nu_score']), dtype=bool)
    NaNMask = np.invert(np.isnan(Data['rec.slc.nu_score']))
    Data['conr.slc.nu_score'][NaNMask] = NuScore(Data['rec.slc.nu_score'][NaNMask])

    # It is useful to define a combined "TPC Preselection" quantity based on the RecoInFV, Pandora
    # not_clear_cosmic, and Pandora nu_score variables.
    Data['conr.slc.tpc_preselection'] = (Data['conr.slc.RecoInFV'] &
                                         Data['conr.slc.not_clear_cosmic'] &
                                         Data['conr.slc.nu_score'])

    # Some boolean tags are being interpreted as int8's, which was a horribly frustrating problem
    # to debug. FORCE EVERYTHING.
    Data['conr.slc.not_clear_cosmic'] = Data['conr.slc.not_clear_cosmic'].astype(bool)
    Data['conr.slc.nu_score'] = Data['conr.slc.nu_score'].astype(bool)
    Data['conr.slc.tpc_preselection'] = Data['conr.slc.tpc_preselection'].astype(bool)

def DefineFlashMatchingCuts(Data):
    # Create boolean tags for each slice to mark them as passing the flash matching score cut.

    # We need to mark slices as passing/failing the flash matching score cut, as defined in
    # Selection.py
    Data['conr.slc.flash_score'] = np.zeros(len(Data['rec.slc.fmatch.score']), dtype=bool)
    NaNMask = np.invert(np.isnan(Data['rec.slc.fmatch.score']))
    Data['conr.slc.flash_score'][NaNMask] = FlashScore(Data['rec.slc.fmatch.score'][NaNMask])

    # Apparently some flash scores are negative, which as I understand it means that there is
    # no score, but is not caught by checking for NaN's as usual. It is helpful to further refine
    # the boolean flag with this information.
    Data['conr.slc.flash_score_nonnegative'] = Data['conr.slc.flash_score']
    NegMask = Data['rec.slc.fmatch.score'][NaNMask] < 0
    Data['conr.slc.flash_score_nonnegative'][NaNMask][NegMask] = False

    # It may be useful to define a tag for scores which "pass" as defined by Selection.py, but are
    # actually negative.
    Data['conr.slc.flash_score_negative'] = Data['conr.slc.flash_score']
    Data['conr.slc.flash_score_negative'][NaNMask][np.invert(NegMask)] = False

    # Some boolean tags are being interpreted as int8's, which was a horribly frustrating problem
    # to debug. FORCE EVERYTHING.
    Data['conr.slc.flash_score'] = Data['conr.slc.flash_score'].astype(bool)
    Data['conr.slc.flash_score_nonnegative'] = Data['conr.slc.flash_score_nonnegative'].astype(bool)
    Data['conr.slc.flash_score_negative'] = Data['conr.slc.flash_score_negative'].astype(bool)

def DefineCalorimetricPIDVariables(Data):
    # The calorimetric PID information allows us to do a lot of refinement to the selection, but we
    # first need to put it into a leverageable format.

    # First we want to define a "bestplane" tag for each of the chi^2 quantities that gives us a
    # shortcut for using this information
    for Chi2 in ['chi2_muon', 'chi2_proton']:
        Data['conr.trk.bestplane.' + Chi2] = Data['rec.slc.reco.trk.chi2pid2.' + Chi2]
        B0 = Data['rec.slc.reco.trk.bestplane'] == 0
        B1 = Data['rec.slc.reco.trk.bestplane'] == 1
        Data['conr.trk.bestplane.' + Chi2][B0] = Data['rec.slc.reco.trk.chi2pid0.' + Chi2][B0]
        Data['conr.trk.bestplane.' + Chi2][B1] = Data['rec.slc.reco.trk.chi2pid0.' + Chi2][B1]

    # It will be useful to tag tracks as containing the majority of the energy and the majority of
    # the energy deposited (visibly) on the planes.
    Data['cont.trk.bestmatch_majority_energy'] = (Data['rec.slc.reco.trk.truth.bestmatch.energy'] / Data['rec.slc.reco.trk.truth.total_deposited_energy']) > 0.5
    Data['cont.trk.bestmatch_majority_planeVisE'] = (Data['rec.slc.reco.trk.truth.bestmatch.energy'] / Data['rec.slc.reco.trk.truth.p.planeVisE']) > 0.5

    # Some boolean tags are being interpreted as int8's, which was a horribly frustrating problem
    # to debug. FORCE EVERYTHING.
    Data['cont.trk.bestmatch_majority_energy'] = Data['cont.trk.bestmatch_majority_energy'].astype(bool)
    Data['cont.trk.bestmatch_majority_planeVisE'] = Data['cont.trk.bestmatch_majority_planeVisE'].astype(bool)

def DefinePrimaryTrack(Data):
    # We can further leverage the calorimetric PID information and other track characteristics to
    # tag tracks as muon track candidates.

    PTrack = PrimaryTracks(Data)
    PTrackInd = ak.JaggedArray.fromcounts((PTrack >=0)*1, PTrack[PTrack >= 0])
    TruePTrack = TruePrimaryTracks(Data)
    TruePTrackInd = ak.JaggedArray.fromcounts((TruePTrack >=0)*1, TruePTrack[TruePTrack >= 0])

    # We can create variables corresponding to each reco.trk quantity, except specifically for
    # primary track candidates, i.e. if reco.trk.YYY contains variable YYY for each track, then
    # we create ptrack.YYY which contains YYY for each primary track candidate.
    Keys = list(Data.keys())
    for k in Keys:
        if k.startswith('rec.slc.reco.trk'):
            SLCKey = k.replace('rec.slc.reco.trk.', 'conr.ptrk.')
            if 'truth' in SLCKey: SLCKey = SLCKey.replace('conr', 'cont')
            Data[SLCKey] = np.empty(Data['conr.slc.nu_score'].shape)
            Data[SLCKey][:] = np.NaN
            IsBool = Data[k][0].dtype == 'bool'
            d = Data[k] + 0 # Copy
            Data[SLCKey][PTrack >= 0] = d[PTrackInd[PTrackInd >= 0]].flatten()
            if IsBool:
                Data[SLCKey] = Data[SLCKey] == 1 # Re-cast as bool if necessary.
    Data['conr.slc.has_ptrk'] = PTrack >= 0

    # It is useful to do PID for these primary track candidates. For pions:
    Data['cont.ptrack_pion'] = np.abs(Data['cont.ptrk.truth.p.pdg']) == 211

    # And for protons:
    Data['cont.ptrack_proton'] = np.abs(Data['cont.ptrk.truth.p.pdg']) == 2212
    
