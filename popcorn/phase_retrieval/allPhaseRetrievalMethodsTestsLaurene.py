#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 13:46:27 2021.

@author: quenot
"""
from saveParameters import saveParameters
from PhaseRetrievalClasses import Phase_Retrieval_Experiment
import time

if __name__ == "__main__":
    
    # Parameters to tune
    studied_case = 'TheoreticalStudy' # name of the experiment we want to work on

    do_LCS=True
    do_MISTII_2=False
    do_MISTII_1=False
    do_MISTI=False
    do_UMPA=False
    do_OF=False
    do_Pavlov=False
    do_XSVT=False
    do_save_parameters=True

    phase_retrieval_experiment=Phase_Retrieval_Experiment(studied_case)

    processing_time={}
    processing_time['LCS']=0
    processing_time['UMPA']=0
    processing_time['OF']=0
    processing_time['Pavlov']=0
    processing_time['MISTI']=0
    processing_time['MISTII_1']=0
    processing_time['MISTII_2']=0
    processing_time['XSVT']=0
    
    if do_LCS:
        time0=time.time()
        phase_retrieval_experiment.process_LCS()
        processing_time['LCS']=time.time()-time0
    if do_UMPA:
        time0=time.time()
        phase_retrieval_experiment.process_UMPA()
        processing_time['UMPA']=time.time()-time0
    if do_OF:
        time0=time.time()
        phase_retrieval_experiment.process_OpticalFlow()
        processing_time['OF']=time.time()-time0
    if do_Pavlov:
        time0=time.time()
        phase_retrieval_experiment.process_Pavlov2020()
        processing_time['Pavlov']=time.time()-time0
    if do_MISTI:
        time0=time.time()
        phase_retrieval_experiment.process_MISTI()
        processing_time['MISTI']=time.time()-time0
    if do_MISTII_1:
        time0=time.time()
        phase_retrieval_experiment.process_MISTII_1()
        processing_time['MISTII_1']=time.time()-time0
    if do_MISTII_2:
        time0=time.time()
        phase_retrieval_experiment.process_MISTII_2()
        processing_time['MISTII_2']=time.time()-time0
    if do_XSVT:
        time0=time.time()
        phase_retrieval_experiment.process_XSVT()
        processing_time['XSVT']=time.time()-time0
        
    print(processing_time)

    if do_save_parameters:
        saveParameters(phase_retrieval_experiment, processing_time)
