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
    """
    return 10.*np.log10(P)
