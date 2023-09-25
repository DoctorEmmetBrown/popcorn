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

def launchPhaseRetrieval(experiment, do):
    """
    launches the phase retrieval algorithms set to true in do

    Args:
        experiment (Object): contains all experiment parameters and methods.
        do (DICT): boolean associated to each phase retrieval method.

    Returns:
        processing_time (float): Processing time of the different algos.

    """
    processing_time={}
    
    for method, to_do in do.items():
        if to_do:
            time0=time.time()
            experiment.process_method(method)
            processing_time[method]=time.time()-time0
            
    return processing_time
    


if __name__ == "__main__":
    
    # Parameters to tune
    studied_case = ' ' # name of the experiment we want to work on
    
    do={}
    do['LCS']=1 
    do['rLCS']=1 
    do['LCS_DF']=0
    do['MISTII_2']=0
    do['MISTII_1']=0
    do['MISTI']=0
    do['UMPA']=0
    do['OF']=0 
    do['Pavlov']=0
    do['XST-XSVT']=1 
    do['rXST-XSVT']=1 
    save_parameters=True

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
    
        if save_parameters:
            saveParameters(phase_retrieval_experiment, processing_time, do)
        
    if phase_retrieval_experiment.tomo:
        outpurFolder0=phase_retrieval_experiment.output_folder
        for iproj in range(phase_retrieval_experiment.proj_to_treat_start,phase_retrieval_experiment.proj_to_treat_end, 1):
            print("\n\n Processing projection:" ,iproj)
            phase_retrieval_experiment.open_Is_Ir_tomo(iproj, phase_retrieval_experiment.number_of_projections)
            phase_retrieval_experiment.preProcessAndPadImages()
            phase_retrieval_experiment.currentProjection=iproj
            processing_time=launchPhaseRetrieval(phase_retrieval_experiment, do)
            
        if save_parameters:
            saveParameters(phase_retrieval_experiment, processing_time, do)
            
            
