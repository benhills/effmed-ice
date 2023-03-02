#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 1 2021

@author: benhills
"""

import numpy as np

"""
Indicatrices for single pole and vertical girdle fabrics based on Matsuoka et al. (2009)
"""

def get_indicatrix(em, fabric, theta, psi):
    """

    """
    if fabric == 'single-pole':
        single_pole_indicatrix(em, theta, psi)
    elif fabric == 'vertical-girdle':
        vertical_girdle_indicatrix(em)
    else:
        raise TypeError('Fabric not recognized; choose from single-pole or vertical-girdle.')

    add_properties(em)


def single_pole_indicatrix(em, theta, psi):
    """

    """

    if not hasattr(em,'theta'):
        em.theta = theta

    if not hasattr(em,'psi'):
        em.psi = psi

    em.psi_ = np.arctan(np.sin(em.theta)*np.sin(em.psi-em.psi_w)/(np.cos(em.theta_w)*np.sin(em.theta) * \
                        np.cos(em.psi-em.psi_w)-np.sin(em.theta_w)*np.cos(em.theta)))
    em.theta_ = np.arccos(np.cos(em.theta_w)*np.cos(em.theta)+np.sin(em.theta_w)*np.sin(em.theta)*np.cos(em.psi-em.psi_w))
    em.psi__ = np.arctan(np.sin(em.theta)*np.sin(em.psi-em.psi_w)/(np.cos(em.theta_w)*np.sin(em.theta) * \
                        np.cos(em.psi-em.psi_w)+np.sin(em.theta_w)*np.cos(em.theta)))
    em.theta__ = np.arccos(np.cos(em.theta_w)*np.cos(em.theta)-np.sin(em.theta_w)*np.sin(em.theta)*np.cos(em.psi-em.psi_w))

    em.m1 = (1-em.chi[0])*em.mr_perp + em.chi[0]*em.mr_par
    em.m2 = (1-em.chi[1])*em.mr_perp + em.chi[1]*em.mr_par
    em.m3 = (1-em.chi[2])*em.mr_perp + em.chi[2]*em.mr_par
    em.m_par = em.m3
    em.m_perp = em.m1
    em.m_1 = em.m_par*em.m_perp/np.sqrt(em.m_par**2.*np.cos(em.theta_)**2. + em.m_perp**2.*np.sin(em.theta_)**2.)
    em.m_2 = em.m_perp

    if em.mc_perp == 0. and em.mc_par == 0.:
        em.mc_1 = 0.
        em.mc_2 = 0.
    else:
        em.mc1 = (1-em.chi[0])*em.mc_perp + em.chi[0]*em.mc_par
        em.mc2 = (1-em.chi[1])*em.mc_perp + em.chi[1]*em.mc_par
        em.mc3 = (1-em.chi[2])*em.mc_perp + em.chi[2]*em.mc_par
        em.mc_par = em.mc3
        em.mc_perp = em.mc1
        em.mc_1 = em.mc_par*em.mc_perp/np.sqrt(em.mc_par**2.*np.cos(em.theta_)**2. + em.mc_perp**2.*np.sin(em.theta_)**2.)
        em.mc_2 = em.mc_perp


def vertical_girdle_indicatrix(em):
    """

    """

    em.m1 = (1-em.chi[0])*em.mr_perp + em.chi[0]*em.mr_par
    em.m2 = (1-em.chi[1])*em.mr_perp + em.chi[1]*em.mr_par
    em.m3 = (1-em.chi[2])*em.mr_perp + em.chi[2]*em.mr_par
    A = (np.cos(em.psi_w)**2./(em.m1**2.)+np.sin(em.psi_w)**2./(em.m2**2.)) *\
        np.cos(em.theta_w)**2. + np.sin(em.theta_w)**2./(em.m3**2.)
    B = -(1./(em.m1**2.)-1./(em.m2**2.))*np.cos(em.theta_w)*np.sin(2.*em.psi_w)
    C = np.sin(em.psi_w)**2./(em.m1**2.)+np.cos(em.psi_w)**2./(em.m2**2.)
    em.m_1 = np.sqrt(2./(A+C+np.sqrt(B**2.+(A-C)**2.)))
    em.m_2 = np.sqrt(2./(A+C-np.sqrt(B**2.+(A-C)**2.)))

    if em.mc_perp == 0. and em.mc_par == 0.:
        em.mc_1 = 0.
        em.mc_2 = 0.
    else:
        em.mc1 = (1-em.chi[0])*em.mc_perp + em.chi[0]*em.mc_par
        em.mc2 = (1-em.chi[1])*em.mc_perp + em.chi[1]*em.mc_par
        em.mc3 = (1-em.chi[2])*em.mc_perp + em.chi[2]*em.mc_par
        A = (np.cos(em.psi_w)**2./(em.mc1**2.) + np.sin(em.psi_w)**2./(em.mc2**2.)) *\
            np.cos(em.theta_w)**2. + np.sin(em.theta_w)**2./(em.mc3**2.)
        B = -(1./(em.mc1**2.)-1./(em.mc2**2.))*np.cos(em.theta_w)*np.sin(2.*em.psi_w)
        C = np.sin(em.psi_w)**2./(em.mc1**2.)+np.cos(em.psi_w)**2./(em.mc2**2.)
        em.mc_1 = np.sqrt(2./(A+C+np.sqrt(B**2.+(A-C)**2.)))
        em.mc_2 = np.sqrt(2./(A+C-np.sqrt(B**2.+(A-C)**2.)))

    if em.psi_w == 0.:
        em.psi_ = 0.
    else:
        em.psi_ = 1/2.*np.arctan(B/(A-C))
    em.psi__ = em.psi_


def add_properties(em):
    """

    """

    # Electrical conductivity
    em.sigma_1 = em.omega*em.eps0*em.mc_1
    em.sigma_2 = em.omega*em.eps0*em.mc_2

    # Propagation constants (Fujita et al., 2006; eq 7)
    em.k_1 = np.sqrt(em.eps0*em.mu0*em.m_1**2.*em.omega**2. +
                     1j*em.mu0*em.sigma_1*em.omega)
    em.k_2 = np.sqrt(em.eps0*em.mu0*em.m_2**2.*em.omega**2. +
                     1j*em.mu0*em.sigma_2*em.omega)
