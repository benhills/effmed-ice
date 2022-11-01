#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 1 2021

@author: benhills
"""

import numpy as np

class indicatrix_model():
    """
    indicatrix model based on Matsuoka et al. (2009)
    """

    def __init__(self,fc):
        self.c = 3e8                    # free space wave speed
        self.eps0 = 8.85418782e-12      # free space permittivity
        self.mu0 = 1.25663706e-6        # free space permeability
        self.epsr = 3.12                # relative permittivity for ice
        self.fc = fc                    # system center frequency
        self.mr_perp=np.sqrt(3.152)     # index of refraction perpendicular to c-axis
        self.mr_par=np.sqrt(3.189)      # index of refraction parallel to c-axis

        self.omega = 2.*np.pi*fc        # angular frequency
        self.lambda0 = self.c/self.fc   # system wavelength in free space
        self.k0 = self.omega/self.c     # system wavenumber

        self.Wt = np.array([1,0])       # transmit polarization
        self.Wr = np.array([1,0])       # receive polarization
        self.S = np.array([[-1,0],[0,-1]])  # scattering matrix

        self.χ=[0,0,1]                  # eigenvalues of the c-axes distribution
        self.fabric='single-pole'       # qualitative fabric type


    def R(psi):
        """
        Reflection matrix

        Parameters
        ----------
        psi: float
                angle
        """
        return np.array([[np.cos(psi),np.sin(psi)],[-np.sin(psi),np.cos(psi)]])


    def T(l):
        """
        Transmission matrix

        Parameters
        ----------
        l: float
                angle
        """

        k = self.c*np.sqrt(self.eps0*self.mu0)
        T1 = np.exp(2j*np.pi*self.m1*k*l*self.f/self.c)
        T2 = np.exp(2j*np.pi*self.m2*k*l*self.f/self.c)

        return np.array([[T1,np.zeros_like(l)],[np.zeros_like(l),T2]])


    def birefringent_losses(theta_w,psi_w,d,theta=0.,psi=0.):
        """

        """

        l = d/np.cos(theta_w)

        if self.fabric == 'single-pole':
            single_pole_indicatrix(theta,psi,theta_w,psi_w)
        elif self.fabric == 'vertical-girdle':
            vertical_girdle_indicatrix(theta_w,psi_w,self.chi)
        else:
            raise TypeError('Fabric not recognized; choose from single-pole or vertical-girdle.')

        Amp = np.matmul(self.Wr,self.R(-self.psi_2))
        Amp = np.matmul(Amp,self.T(self.m1_2,self.m2,l,self.fc))
        Amp = np.matmul(Amp,self.R(self.psi_2))
        Amp = np.matmul(Amp,self.S)
        Amp = np.matmul(Amp,self.R(-self.psi_1))
        Amp = np.matmul(Amp,self.T(self.m1_1,self.m2,l,self.fc))
        Amp = np.matmul(Amp,self.R(self.psi_1))
        Amp = np.matmul(Amp,np.transpose(self.Wt))

        return np.real(10.*np.log10(Amp**2.))


    def single_pole_indicatrix(theta,psi,thetaw,psiw):
        """

        """

        self.m1 = (1-self.chi[0])*self.mr_perp + self.chi[0]*self.mr_par
        self.m2 = (1-self.chi[1])*self.mr_perp + self.chi[1]*self.mr_par
        self.m3 = (1-self.chi[2])*self.mr_perp + self.chi[2]*self.mr_par
        self.m_par = self.m3
        self.m_perp = self.m1

        self.psi_1 = np.arctan(np.sin(θ)*np.sin(ψ-ψw)/(np.cos(θw)*np.sin(θ)*np.cos(ψ-ψw)-np.sin(θw)*np.cos(θ)))
        self.theta_1 = np.arccos(np.cos(θw)*np.cos(θ)+np.sin(θw)*np.sin(θ)*np.cos(ψ-ψw))
        self.psi_2 = np.arctan(np.sin(θ)*np.sin(ψ-ψw)/(np.cos(θw)*np.sin(θ)*np.cos(ψ-ψw)+np.sin(θw)*np.cos(θ)))
        self.theta_2 = np.arccos(np.cos(θw)*np.cos(θ)-np.sin(θw)*np.sin(θ)*np.cos(ψ-ψw))

        self.m1_1 = self.m_par*self.m_perp/np.sqrt(self.m_par**2.*np.cos(θ_1)**2. + self.m_perp**2.*np.sin(θ_1)**2.)
        self.m1_2 = self.m_par*self.m_perp/np.sqrt(self.m_par**2.*np.cos(θ_2)**2. + self.m_perp**2.*np.sin(θ_2)**2.)
        self.m2 = self.m_perp


    def vertical_girdle_indicatrix(theta_w,psi_w):
        """

        """

        self.m1 = (1-self.chi[0])*self.mr_perp + self.chi[0]*self.mr_par
        self.m2 = (1-self.chi[1])*self.mr_perp + self.chi[1]*self.mr_par
        self.m3 = (1-self.chi[2])*self.mr_perp + self.chi[2]*self.mr_par

        A = (np.cos(psi_w)**2./(self.m1**2.)+np.sin(psi_w)**2./(self.m2**2.))*np.cos(theta_w)**2.+np.sin(theta_w)**2./(self.m3**2.)
        B = -(1./(self.m1**2.)-1./(self.m2**2.))*np.cos(theta_w)*np.sin(2.*psi_w)
        C = np.sin(psi_w)**2./(self.m1**2.)+np.cos(psi_w)**2./(self.m2**2.)

        self.psi_1 = 1/2.*np.arctan(B/(A-C))
        self.m1_1 = np.sqrt(2./(A+C+np.sqrt(B**2.+(A-C)**2.)))
        self.m2 = np.sqrt(2./(A+C-np.sqrt(B**2.+(A-C)**2.)))

        self.psi_2 = self.psi_1
        self.m1_2 = self.m1_1
