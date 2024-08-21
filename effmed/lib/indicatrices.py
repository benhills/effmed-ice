#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 1 2021

@author: benhills
"""

import numpy as np
from np import matmul

"""
Indicatrices for single pole and vertical girdle fabrics based on Matsuoka et al. (2009)
"""

def get_prop_const(em, indicatrix, theta, psi, prop_up):
    """
    Get Propagation Constants
    Use the COF indicatrix to get the conductivity, sigma, and propagation constant, k.

    Parameters
    ----------
    em:     class,  effective medium model class
    indicatrix:      str,    qualitative fabric 'type' for indicatrix selection
    theta:      float,  polar angle of vertical eigenvector (chi[2])
    psi:        float,  azimuthal angle of vertical eigenvalue or girdle
    prop_up:    bool,   propagate up? changes the indicatrix output
    """

    # Get real and imaginary refractive index along crystal orientation fabric (disregard propagation direction for now)
    em.mr = (1.-em.chi)*em.mr_perp + em.chi*em.mr_par
    em.mi = (1.-em.chi)*em.mi_perp + em.chi*em.mi_par

    # Rotate refractive index based on wave propagation direction
    if indicatrix == 'uniaxial':
        uniaxial_indicatrix(em, theta, psi, prop_up)
    elif indicatrix == 'biaxial':
        biaxial_indicatrix(em, psi)
    else:
        raise TypeError('Indicatrix not recognized; choose from uniaxial or biaxial.')

    # Get conductivity and propagation constant from refractive index (Fujita et al., 2006; eq 7)
    em.sigma = em.omega*em.eps0*em.mi_**2. # conductivity
    em.k = np.sqrt(em.eps0*em.mu0*em.mr_**2.*em.omega**2. +
                     1j*em.mu0*em.sigma*em.omega) # propagation constant


def uniaxial_indicatrix(em, theta, psi, prop_up):
    """
    Get the indicatrix for a Single Pole fabric
    Matsuoka 2009 Appendix IIA

    Parameters
    ----------
    em:         class,  effective medium model class
    theta:      float,  polar angle of vertical eigenvector (chi[2])
    psi:        float,  azimuthal angle of vertical eigenvalue or girdle
    prop_up:    bool,   propagate up? changes the indicatrix output
    """

    # Get rotation angles for downward (_) and upward (__) propagation
    em.psi_ = np.arctan(np.sin(theta)*np.sin(em.psi_w-psi)/(np.cos(em.theta_w)*np.sin(theta) * \
                        np.cos(em.psi_w-psi)-np.sin(em.theta_w)*np.cos(theta)))
    em.theta_ = np.arccos(np.cos(em.theta_w)*np.cos(theta)+np.sin(em.theta_w)*np.sin(theta)*np.cos(em.psi_w-psi))
    em.psi__ = np.arctan(np.sin(theta)*np.sin(em.psi_w-psi)/(np.cos(em.theta_w)*np.sin(theta) * \
                        np.cos(em.psi_w-psi)+np.sin(em.theta_w)*np.cos(theta)))
    em.theta__ = np.arccos(np.cos(em.theta_w)*np.cos(theta)-np.sin(em.theta_w)*np.sin(theta)*np.cos(em.psi_w-psi))

    # Rotate index of refraction
    if prop_up:
        em.mr_ = np.array([em.mr[2]*em.mr[0]/np.sqrt(em.mr[2]**2.*np.cos(em.theta__)**2. + em.mr[0]**2.*np.sin(em.theta__)**2.),
                           em.mr[0]])
    else:
        em.mr_ = np.array([em.mr[2]*em.mr[0]/np.sqrt(em.mr[2]**2.*np.cos(em.theta_)**2. + em.mr[0]**2.*np.sin(em.theta_)**2.),
                           em.mr[0]])
    if np.all(em.mi==0.):
        em.mi_ = np.array([0.,0.])
    else:
        if prop_up:
            em.mi_ = np.array([em.mi[2]*em.mi[0]/np.sqrt(em.mi[2]**2.*np.cos(em.theta__)**2. + em.mi[0]**2.*np.sin(em.theta__)**2.),
                               em.mi[0]])
        else:
            em.mi_ = np.array([em.mi[2]*em.mi[0]/np.sqrt(em.mi[2]**2.*np.cos(em.theta_)**2. + em.mi[0]**2.*np.sin(em.theta_)**2.),
                               em.mi[0]])


def biaxial_indicatrix(em, psi=0., tol = 1e-10):
    """
    Get the indicatrix for a Vertical Girdle fabric
    Matsuoka 2009 Appendix IIB

    Parameters
    ----------
    em:         class,  effective medium model class
    psi:        float,  azimuthal angle of vertical eigenvalue or girdle
    tol:        float,  tolerance for cutting off psi_ = 0
    """

    # Update real index of refraction with function for intersection ellipse (A, B, C)
    A = (np.cos(em.psi_w-psi)**2./(em.mr[0]**2.)+np.sin(em.psi_w-psi)**2./(em.mr[1]**2.)) *\
        np.cos(em.theta_w)**2. + np.sin(em.theta_w)**2./(em.mr[2]**2.)
    B = -(1./(em.mr[0]**2.)-1./(em.mr[1]**2.))*np.cos(em.theta_w)*np.sin(2.*(em.psi_w-psi))
    C = np.sin(em.psi_w-psi)**2./(em.mr[0]**2.)+np.cos(em.psi_w-psi)**2./(em.mr[1]**2.)
    em.mr_ = np.array([np.sqrt(2./(A+C+np.sign(A-C)*np.sqrt(B**2.+(A-C)**2.))),
                        np.sqrt(2./(A+C-np.sign(A-C)*np.sqrt(B**2.+(A-C)**2.)))])

    # Update imaginary
    if np.all(em.mi==0.):
        em.mi_ = np.array([0.,0.])
    else:
        A = (np.cos(em.psi_w-psi)**2./(em.mi[0]**2.) + np.sin(em.psi_w-psi)**2./(em.mi[1]**2.)) *\
            np.cos(em.theta_w)**2. + np.sin(em.theta_w)**2./(em.mi[2]**2.)
        B = -(1./(em.mi[0]**2.)-1./(em.mi[1]**2.))*np.cos(em.theta_w)*np.sin(2.*(em.psi_w-psi))
        C = np.sin(em.psi_w-psi)**2./(em.mi[0]**2.)+np.cos(em.psi_w-psi)**2./(em.mi[1]**2.)
        em.mi_ = np.array([np.sqrt(2./(A+C+np.sign(A-C)*np.sqrt(B**2.+(A-C)**2.))),
                            np.sqrt(2./(A+C-np.sign(A-C)*np.sqrt(B**2.+(A-C)**2.)))])

    # Rotate the azimuthal angle with the intersection ellipse
    if abs(em.psi_w-psi) == 0.:
        em.psi_ = 0.
    elif abs(B) < tol and abs(A-C) < tol:
        em.psi_ = 0.
    else:
        em.psi_ = 1/2.*np.arctan(B/(A-C))
    em.psi__ = em.psi_


def rotate_biaxial(rotation):
    """
    Rotate the biaxial indicatrix
    For a generalized girdle fabric (doesn't need to be vertical)
    Matsuoka 2009 Appendix IIC

    Parameters
    ----------
    rotation:   3x1 array,
    """

    phi,theta,psi = rotation
    M1 = np.array([[np.cos(phi), -np.sin(phi), 0.],
                   [np.sin(phi), np.cos(phi), 0.],
                   [0., 0., 1.]])
    M2 = np.array([[1, 0., 0.],
                   [0., np.cos(theta), -np.sin(theta)],
                   [0., np.sin(theta), np.cos(theta)]])
    M3 = np.array([[np.cos(psi), -np.sin(psi), 0.],
                   [np.sin(psi), np.cos(psi), 0.],
                   [0., 0., 1.]])

    Mtilt = matmul(matmul(M1,M2),M3)
