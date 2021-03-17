#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 13:46:27 2021.

@author: quenot
"""
from saveParameters import saveParameters
from PhaseRetrievalClasses import Phase_Retrieval_Experiment

if __name__ == "__main__":
    # Parameters to tune
    studied_case = 'MoucheSimapAout2017' # name of the experiment we want to work on

    do_LCS=True
    do_MISTII_2=True
    do_MISTII_1=True
    do_MISTI=True
    do_UMPA=True
    do_OF=True
    do_Pavlov=True
    do_save_parameters=False

    phase_retrieval_experiment=Phase_Retrieval_Experiment(studied_case)

    if do_LCS:
        phase_retrieval_experiment.process_LCS()
    if do_UMPA:
        phase_retrieval_experiment.process_UMPA()
    if do_OF:
        phase_retrieval_experiment.process_OpticalFlow()
    if do_Pavlov:
        phase_retrieval_experiment.process_Pavlov2020()
    if do_MISTI:
        phase_retrieval_experiment.process_MISTI()
    if do_MISTII_1:
        phase_retrieval_experiment.process_MISTII_1()
    if do_MISTII_2:
        phase_retrieval_experiment.process_MISTII_2()

    if do_save_parameters:
        saveParameters(phase_retrieval_experiment)
