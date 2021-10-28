#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 13:46:27 2021.

@author: quenot
"""
from saveParameters import saveParameters
from PhaseRetrievalClasses import Phase_Retrieval_Experiment
import time
import datetime
import os

def launchPhaseRetrieval(phase_retrieval_experiment, do):
    
    processing_time={}
    processing_time['LCS']=0
    processing_time['UMPA']=0
    processing_time['OF']=0
    processing_time['Pavlov']=0
    processing_time['MISTI']=0
    processing_time['MISTII_1']=0
    processing_time['MISTII_2']=0
    processing_time['XSVT']=0
    
    if do['LCS']:
        time0=time.time()
        phase_retrieval_experiment.process_LCS()
        processing_time['LCS']=time.time()-time0
    if do['UMPA']:
        time0=time.time()
        phase_retrieval_experiment.process_UMPA()
        processing_time['UMPA']=time.time()-time0
    if do['OF']:
        time0=time.time()
        phase_retrieval_experiment.process_OpticalFlow()
        processing_time['OF']=time.time()-time0
    if do['Pavlov']:
        time0=time.time()
        phase_retrieval_experiment.process_Pavlov2020()
        processing_time['Pavlov']=time.time()-time0
    if do['MISTI']:
        time0=time.time()
        phase_retrieval_experiment.process_MISTI()
        processing_time['MISTI']=time.time()-time0
    if do['MISTII_1']:
        time0=time.time()
        phase_retrieval_experiment.process_MISTII_1()
        processing_time['MISTII_1']=time.time()-time0
    if do['MISTII_2']:
        time0=time.time()
        phase_retrieval_experiment.process_MISTII_2()
        processing_time['MISTII_2']=time.time()-time0
    if do['XSVT']:
        time0=time.time()
        phase_retrieval_experiment.process_XSVT()
        processing_time['XSVT']=time.time()-time0
        
    return processing_time
    


if __name__ == "__main__":
    
    # Parameters to tune
    studied_case = 'Tomo_md1217' # name of the experiment we want to work on
    
    do={}
    do['LCS']=False
    do['MISTII_2']=False
    do['MISTII_1']=False
    do['MISTI']=True
    do['UMPA']=False
    do['OF']=False
    do['Pavlov']=False
    do['XSVT']=False
    do['save_parameters']=True

    phase_retrieval_experiment=Phase_Retrieval_Experiment(studied_case, do)
    # We create a folder for each retrieval test
    now=datetime.datetime.now()
    phase_retrieval_experiment.expID=now.strftime("%Y%m%d-%H%M%S") #
    phase_retrieval_experiment.output_folder+=phase_retrieval_experiment.expID
    os.mkdir(phase_retrieval_experiment.output_folder)
    
    if not phase_retrieval_experiment.tomo:
        phase_retrieval_experiment.open_Is_Ir()
        phase_retrieval_experiment.preProcessAndPadImages()
        processing_time=launchPhaseRetrieval(phase_retrieval_experiment,do)
        print(processing_time)
    
        if do['save_parameters']:
            saveParameters(phase_retrieval_experiment, processing_time, do)
        
    if phase_retrieval_experiment.tomo:
        outpurFolder0=phase_retrieval_experiment.output_folder
        for iproj in range(phase_retrieval_experiment.proj_to_treat_start,phase_retrieval_experiment.proj_to_treat_end, 1):
            print("\n\n Processing projection:" ,iproj)
            phase_retrieval_experiment.open_Is_Ir_tomo(iproj, phase_retrieval_experiment.number_of_projections)
            phase_retrieval_experiment.preProcessAndPadImages()
            phase_retrieval_experiment.currentProjection=iproj
            processing_time=launchPhaseRetrieval(phase_retrieval_experiment, do)
            
        if do['save_parameters']:
            saveParameters(phase_retrieval_experiment, processing_time, do)
            
            
