#!/usr/bin/env python3
# -*- coding: utf-8 -*-
######################################################
# B0 -> rho0 gamma reco
######################################################

import os, sys, glob
from basf2 import *
from modularAnalysis import *
from variables import variables
import steeringhelper, getKEKCCfiles
from stdPhotons import *
from stdPi0s import *
from stdCharged import *
from flavorTagger import *

MC7_base = '/ghi/fs01/belle2/bdata/MC/release-00-07-02/DBxxxxxxxx/MC7'
mode = 'B02rho0gamma'
B_charge = 0
training_types =['signal', 'generic', 'qqbar', 'X_s', 'X_d', 'custom', 'uu', 'dd', 'cc', 'ss', 'kstar']

if (len(sys.argv) < 3 or sys.argv[1] not in training_types):
    sys.exit("Usage:\n\tbasf2 <FILENAME> <steering_file> <signal, generic, qqbar, X_s, X_d, custom> <MEL, KEKCC>")
step = str(sys.argv[1])
inputFile = str(sys.argv[2])

# Begin reconstruction
rootOutputFile = mode + step + inputFile.split('_')[1] + '.root'
# rootOutputFile = mode + step + '.root'
ext = 0
while os.path.isfile(rootOutputFile):
    ext += 1
    rootOutputFile = mode + step + str(ext) + '.root'
inputMdstList('default', [inputFile])

# Get final state particles
fillParticleList('pi+:std', 'piid > 0.5 and Kid < 0.9 and p > 0.1 and chiProb > 0.001 and abs(dz) < 4.0 and abs(dr) < 0.2', True)
fillParticleList('K+:std', 'Kid > 0.5 and piid < 0.9 and p > 0.1 and chiProb > 0.001 and abs(dz) < 4.0 and abs(dr) < 0.2', True)

# E_gamma > 1.6 GeV - mandated by minimum theoretical predications
fillParticleList('gamma:all', 'goodGamma == 1')
cutAndCopyList('gamma:highE', 'gamma:all', '1.5 < useCMSFrame(E) < 3.5 and 0.8 < clusterE9E25')
matchMCTruth('gamma:highE')

# reconstruct rho -> pi+ pi- decay
# rho resonance quite broad, cut on the lower bound of rho(770)
reconstructDecay('rho0:all -> pi+:std pi-:std', '0.55 < M')
vertexRave('rho0:all', 0.001)
applyCuts('rho0:all', '2.3 < useCMSFrame(p) < 3.3') #'2.3 < useCMSFrame(p) < 2.9') #'0.6 < M < 1.0
matchMCTruth('rho0:all')

# reconstruct B0 -> rho0 gamma decay
reconstructDecay('B0:sig -> rho0:all gamma:highE', '5.2 < Mbc < 5.3 and abs(deltaE) < 1.0')
vertexRave('B0:sig', 0.001)
matchMCTruth('B0:sig')

# Train only on correctly reconstructed events
if step not in ['generic','X_s', 'X_d', 'custom']:
    applyCuts('B0:sig', 'formula(isContinuumEvent+isSignal)>0.5')

buildRestOfEvent('B0:sig')

# Flavour tagging
flavorTagger(
    particleList='B0:sig',
    mode='Expert',
    weightFiles='B2JpsiKs_mu',
    combinerMethods=['TMVA-FBDT', 'FANN-MLP'],
    #workingDirectory = '/group/belle2/tutorial/2017_Feb11/Advanced/PhiKs/',
    workingDirectory=os.environ['BELLE2_LOCAL_DIR'] + '/analysis/data',
    belleOrBelle2='Belle2')

# Perform pi0 veto, see module for details
steeringhelper.pi0_eta_veto('B0:sig', 'rho0:all')

# get tag vertex, continuum suppression
TagV('B0:sig', 'breco')
buildContinuumSuppression('B0:sig')

# Save physics data objects to uDST for later analysis
steeringhelper.mySkimOutputUdst(os.path.join('udst', mode+step), ['B0:sig'])

# create and fill flat Ntuple with MCTruth and kinematic information
toolsB0 = ['CustomFloats[isSignal]', '^B0 -> [^rho0 -> ^pi+ ^pi-] ^gamma']
# toolsB0 += ['MCHierarchy', 'B0 -> ^rho0 ^gamma']
# toolsB0 += ['CustomFloats[nDaughters]', '^B0']
# toolsB0 += ['CustomFloats[daughter(0,PDG)]', '^B0']
# toolsB0 += ['CustomFloats[daughter(1,PDG)]', '^B0']
# toolsB0 += ['InvMass', 'B0 -> ^rho0 gamma']
# toolsB0 += ['DeltaEMbc', '^B0']
# toolsB0 += ['Kinematics', '^B0 -> ^rho0 ^gamma']

cluster_vars = [
        'clusterE9E21', 'clusterHighestE', 'clusterLAT',
        'clusterNHits', 'clusterPhi', 'clusterR', 'clusterTheta',
        'clusterTiming', 'clusterReg', 'clusterUncorrE',
        'minC2HDist', 'ECLEnergy', 'nECLClusters']

new_vars = ['nTracks', 'missingMomentum', 'missingMass', 'daughterSumOf(pt)']

gamma_vars = ['useCMSFrame(p)', 'useCMSFrame(cosTheta)',
        'useCMSFrame(phi)', 'useCMSFrame(E)', 'm2Recoil',
        'eRecoil', 'pRecoil'] + cluster_vars

B_gamma_vars = ['rapidity', 'pseudoRapidity']
B_gamma_NTupleMaker_vars = ['MomentumVectorDeviation', 'Helicity']

# Save photon features to NTuple
toolsGamma = ['CustomFloats[isSignal]', '^B0 -> rho0 ^gamma']
for var in gamma_vars:
    toolsGamma += ['CustomFloats['+var+']', 'B0 -> rho0 ^gamma']
for var in B_gamma_vars:
    toolsGamma += ['CustomFloats['+var+']', '^B0 -> rho0 ^gamma']
for var in B_gamma_NTupleMaker_vars:
    toolsGamma += [var, '^B0 -> rho0 ^gamma']
toolsGamma += ['CMSKinematics', 'B0 -> rho0 ^gamma']
toolsGamma += ['CustomFloats[extraInfo(pi0veto)]', '^B0']
toolsGamma += ['CustomFloats[extraInfo(etaveto)]', '^B0']

for var in cluster_vars:
    toolsB0 += ['CustomFloats['+var+']', 'B0 -> rho0 ^gamma']
for var in new_vars:
    toolsB0 += ['CustomFloats['+var+']', '^B0 -> rho0 gamma']


# Continuum variables
toolsB0 += ['ContinuumSuppression', '^B0']
# toolsB0 += ['CustomFloats[isNotContinuumEvent]', '^B0']

# Rest of Event
toolsB0 += ['CustomFloats[q2Bh]', '^B0']
toolsB0 += ['DeltaT', '^B0']
toolsB0 += ['CustomFloats[DeltaB]', '^B0']
toolsB0 += ['CustomFloats[DeltaZ]', '^B0']

# Properties of signal B and daughters, mostly vertex + kinematic
toolsB0 += ['DeltaEMbc', '^B0']
toolsB0 += ['CMSKinematics', '^B0 -> [^rho0 -> ^pi+ ^pi-]  ^gamma']
toolsB0 += ['CustomFloats[useCMSFrame(p)]', '^B0 -> ^rho0 ^gamma']
toolsB0 += ['CustomFloats[useCMSFrame(cosTheta)]', '^B0 -> ^rho0 ^gamma']
toolsB0 += ['CustomFloats[useCMSFrame(phi)]', '^B0 -> ^rho0 ^gamma']
toolsB0 += ['CustomFloats[useCMSFrame(E)]', 'B0 -> ^rho0 ^gamma']
toolsB0 += ['CustomFloats[useRestFrame(daughter(0,E))]', '^B0 -> rho0 gamma']
toolsB0 += ['CustomFloats[useRestFrame(daughter(1,E))]', '^B0 -> rho0 gamma']
toolsB0 += ['CustomFloats[InvM]', '^B0 -> ^rho0 ^gamma']
toolsB0 += ['CustomFloats[decayAngle(0)]', '^B0']
toolsB0 += ['CustomFloats[decayAngle(1)]', '^B0']
toolsB0 += ['MomentumVectorDeviation', '^B0 -> ^rho0 gamma']
toolsB0 += ['MomentumVectorDeviation', '^B0 -> rho0 ^gamma']
toolsB0 += ['CustomFloats[chiProb]', '^B0']
toolsB0 += ['CustomFloats[daughterAngle(0,1)]', '^B0 -> rho0 gamma']
toolsB0 += ['CustomFloats[daughterInvariantMass(0,1)]', '^B0 -> rho0 gamma']
# Vertex (-ish)
toolsB0 += ['Vertex', '^B0'] # VtxPValue only
toolsB0 += ['CustomFloats[dr]', '^B0']
toolsB0 += ['CustomFloats[dz]', '^B0']
toolsB0 += ['CustomFloats[pRecoil]', '^B0']
toolsB0 += ['CustomFloats[eRecoil]', '^B0']
toolsB0 += ['CustomFloats[m2Recoil]', '^B0']
toolsB0 += ['CustomFloats[isSignal:extraInfo(pi0veto)]', '^B0']
toolsB0 += ['CustomFloats[extraInfo(etaveto)]', '^B0']
toolsB0 += ['CustomFloats[useRestFrame(daughter(0, p))]', '^B0 -> rho0 gamma']
toolsB0 += ['CustomFloats[useRestFrame(daughter(0, cosTheta))]', '^B0 -> rho0 gamma']
toolsB0 += ['CustomFloats[useRestFrame(daughter(0, phi))]', '^B0 -> rho0 gamma']
toolsB0 += ['CustomFloats[helicityAngle(0,0)]', '^B0 -> [rho0 -> pi+ pi-] gamma']
toolsB0 += ['CustomFloats[helicityAngle(0,1)]', '^B0 -> [rho0 -> pi+ pi-] gamma']
toolsB0 += ['CustomFloats[pseudoRapidity]', '^B0 -> [^rho0 -> pi+ pi-] ^gamma']
toolsB0 += ['CustomFloats[rapidity]', '^B0 -> [^rho0 -> pi+ pi-] ^gamma']
toolsB0 += ['Helicity', '^B0 -> [^rho0 -> pi+ pi-] gamma']


# Variables for BDT-aided identification of intermediate resonances (and K*+)
toolsRes = ['CustomFloats[isSignal]', '^B0 -> ^rho0 gamma']
toolsRes += ['Vertex', 'B0 -> ^rho0 gamma'] # VtxPValue only
toolsRes += ['CustomFloats[InvM]', 'B0 -> ^rho0 gamma']
toolsRes += ['CustomFloats[chiProb]', 'B0 -> ^rho0 gamma']
toolsRes += ['CustomFloats[ImpactXY]', 'B0 -> ^rho0 gamma']
toolsRes += ['CustomFloats[daughterAngle(0,1)]', 'B0 -> ^rho0 gamma']
toolsRes += ['CustomFloats[daughterInvariantMass(0,1)]', 'B0 -> ^rho0 gamma']
toolsRes += ['MomentumVectorDeviation', 'B0 -> [^rho0 -> ^pi+ pi-] gamma']
toolsRes += ['MomentumVectorDeviation', 'B0 -> [^rho0 -> pi+ ^pi-] gamma']
toolsRes += ['CMSKinematics', 'B0 -> [^rho0 -> ^pi+ ^pi-]  gamma']
toolsRes += ['CustomFloats[useCMSFrame(cosTheta)]', 'B0 -> [^rho0 -> ^pi+ ^pi-] gamma']
toolsRes += ['CustomFloats[useCMSFrame(phi)]', 'B0 -> [^rho0 -> ^pi+ ^pi-] gamma']
toolsRes += ['CustomFloats[decayAngle(0)]', 'B0 -> ^rho0 gamma']
toolsRes += ['CustomFloats[decayAngle(1)]', 'B0 -> ^rho0 gamma']
toolsRes += ['TrackHits', 'B0 -> [rho0 -> ^pi+ ^pi-] gamma']
toolsRes += ['CustomFloats[dr]', 'B0 -> [^rho0 -> ^pi+ ^pi-] gamma']
toolsRes += ['CustomFloats[dz]', 'B0 -> [^rho0 -> ^pi+ ^pi-] gamma']
toolsRes += ['CustomFloats[pValue]','B0 -> [rho0 -> ^pi+ ^pi-] gamma']
toolsRes += ['CustomFloats[VertexZDist]',  'B0 -> ^rho0 gamma']
toolsRes += ['CustomFloats[significanceOfDistance]', 'B0 -> ^rho0 gamma']
toolsRes += ['CustomFloats[pRecoil]', 'B0 -> ^rho0 gamma']
toolsRes += ['CustomFloats[eRecoil]', 'B0 -> ^rho0 gamma']
toolsRes += ['CustomFloats[m2Recoil]', 'B0 -> ^rho0 gamma']
#toolsRes += ['CustomFloats[Kid]', 'K*+ -> ^rho0 gamma']
toolsRes += ['CustomFloats[piid]', 'B0 -> [rho0 -> ^pi+ ^pi-] gamma']
toolsRes += ['Helicity', '^B0 -> [^rho0 -> pi+ pi-] gamma']
toolsRes += ['Helicity', 'B0 -> [rho0 -> ^pi+ ^pi-] gamma']
toolsRes += ['CustomFloats[rapidity]', 'B0 -> [^rho0 -> pi+ pi-] gamma']
toolsRes += ['CustomFloats[pseudoRapidity]', 'B0 -> [^rho0 -> pi+ pi-] gamma']

# write out the flat ntuple
ntupleFile(rootOutputFile)
ntupleTree('b0', 'B0:sig', toolsB0)
ntupleTree('res', 'B0:sig', toolsRes)
ntupleTree('gamma', 'B0:sig', toolsGamma)

# Process the events
summaryOfLists(['B0:sig', 'gamma:highE'])
process(analysis_main)

# print out the summary
print(statistics)
