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


def rotational_transform(em, theta_start=0, theta_end=np.pi, n_thetas=100):
    """
    Azimuthal (rotational) shift of principal axes
    to create a 2-d depth-azimuth image for all four polarizations
    Mott, 2006

    Parameters
    ---------
    self: class
        effective medium model object
    theta_start: float
        Starting point for array of azimuths
    theta_end: float
        Ending point for array of azimuths
    n_thetas: int
        number of thetas to rotate through
    """

    em.thetas = np.linspace(theta_start, theta_end, n_thetas)

    em.HH = np.empty((len(em.range), len(em.thetas))).astype(np.cdouble)
    em.HV = np.empty((len(em.range), len(em.thetas))).astype(np.cdouble)
    em.VH = np.empty((len(em.range), len(em.thetas))).astype(np.cdouble)
    em.VV = np.empty((len(em.range), len(em.thetas))).astype(np.cdouble)

    for i, theta in enumerate(em.thetas):
        em.HH[:, i] = em.shh*np.cos(theta)**2. + \
            (em.svh + em.shv)*np.sin(theta)*np.cos(theta) + \
            em.svv*np.sin(theta)**2
        em.HV[:, i] = em.shv*np.cos(theta)**2. + \
            (em.svv - em.shh)*np.sin(theta)*np.cos(theta) - \
            em.svh*np.sin(theta)**2
        em.VH[:, i] = em.svh*np.cos(theta)**2. + \
            (em.svv - em.shh)*np.sin(theta)*np.cos(theta) - \
            em.shv*np.sin(theta)**2
        em.VV[:, i] = em.svv*np.cos(theta)**2. - \
            (em.svh + em.shv)*np.sin(theta)*np.cos(theta) + \
            em.shh*np.sin(theta)**2


def coherence(em):
    """
    Phase correlation between two elements of the scattering matrix
    Jodan et al. (2019) eq. 13

    Parameters
    ---------
    em: class
        effective medium model object
    """

    em.chhvv = np.empty_like(em.HH)
    for i in range(len(em.range)):
        for j in range(len(em.thetas)):
            s1 = em.HH[i,j]
            s2 = em.VV[i,j]
            top = np.dot(s1, np.conj(s2))
            bottom = np.sqrt(np.abs(s1)**2.*np.abs(s2)**2.)
            em.chhvv[i,j] = top/bottom


def phase_gradient2d(em):
    """
    Depth-gradient of hhvv phase image.
    Jordan et al. (2019) eq 23

    Parameters
    ---------
    em: class
        effective medium model object
    """

    # Real and imaginary parts of the hhvv coherence
    R_ = np.real(em.chhvv).copy()
    I_ = np.imag(em.chhvv).copy()

    # Depth gradient for each component
    dRdz = np.gradient(R_, em.range, axis=0)
    dIdz = np.gradient(I_, em.range, axis=0)

    # Phase-depth gradient from Jordan et al. (2019) eq. 23
    em.dphi_dz = (R_*dIdz-I_*dRdz)/(R_**2.+I_**2.)
