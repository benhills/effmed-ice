#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 1 2023

@author: benhills
"""

import numpy as np

"""
Supplemental functions for the effective medium model
"""

def dB(P):
    """
    Convert power to decibels

    Parameters
    ----------
    P:  float,  Power
    """
    return 10.*np.log10(P)


def fresnel(chis):
    """
    Get reflection ratio from eigenvalues

    Parameters
    ----------
    chis: 3xN array, COF eigenvalues
    """

    top = np.diff(chis[:,1])
    bot = np.diff(chis[:,0])
    f = np.sqrt((top/bot)**2.)

    return np.insert(f,0,1.)
