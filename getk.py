#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 09:16:10 2020

@author: quenot
"""

import numpy as np


def getk(energy):
    """
    energy in eV
    """
    h=6.626e-34
    c=2.998e8
    e=1.6e-19
    k=2*np.pi*energy*e/(h*c)
    return k



     
if __name__ == "__main__":
    K=getk(25000)
    print("k=", K)
