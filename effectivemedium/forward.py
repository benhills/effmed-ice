#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 1 2021

@author: benhills
"""

import numpy as np
from numpy import matmul

class effective_medium():
    """
    """

    def __init__(self):
        self.c = 3e8
        self.eps0 = 8.85418782e-12
        self.epsr = 3.12
        self.mu0 = 1.25663706e-6

    def radar_constants(self,fc):
        """
        """
        self.fc = fc
        self.omega = 2.*np.pi*fc
        self.lambda0 = self.c/self.fc
        self.k0 = self.omega/self.c

    def delta_epsilon(self,Temp):
        """
        """
        self.deps = 0.0256 + 3.57e-5*Temp

    def reflection(self,gammax,gammay):
        """
        """
        self.G = np.array([[gammax,0],[0,gammay]])

    def rotation(self,theta):
        """
        """
        self.R = np.array([[np.cos(theta),-np.sin(theta)],
                           [np.sin(theta),np.cos(theta)]])

    def attenuation(self,z):
        """
        """
        self.D = np.exp(1j*self.k0*z)/(4.*np.pi*z)

    def transmission(self,dz,epsx,epsy,eps2x=0.,eps2y=0.):
        """
        """

        sigmax = self.omega*self.eps0*eps2x
        sigmay = self.omega*self.eps0*eps2y

        kx = np.sqrt(self.eps0*self.mu0*epsx*self.omega**2.+\
                1j*self.mu0*sigmax*self.omega)
        ky = np.sqrt(self.eps0*self.mu0*epsy*self.omega**2.+\
                1j*self.mu0*sigmay*self.omega)

        Tx = np.exp(-1j*self.k0*dz+1j*kx*dz)
        Ty = np.exp(-1j*self.k0*dz+1j*ky*dz)

        self.T = np.array([[Tx, 0],
                           [0, Ty]])

    def scattering_matrix(self,z,dz,thetas,epsxs,epsys,D=None):
        """
        """

        N = int(z//dz)
        if type(thetas) is float:
            thetas = thetas*np.ones(N+1)
        if type(epsxs) is float:
            epsxs = epsxs*np.ones(N+1)
        if type(epsys) is float:
            epsys = epsys*np.ones(N+1)

        Prop_down = np.eye(2).astype(np.complex)
        Prop_up = np.eye(2).astype(np.complex)
        for layer in range(N):
            self.rotation(thetas[layer])
            self.transmission(dz,epsxs[layer],epsys[layer])
            # Update the electrical field downward propagation to the scattering interface
            Prop_down = matmul(Prop_down,matmul(matmul(self.R,self.T),np.transpose(self.R)))
            # Update upward propagation
            Prop_up = matmul(Prop_up,matmul(matmul(self.R,self.T),np.transpose(self.R)))

        if z%dz > 0:
            self.rotation(thetas[N])
            self.transmission(z%dz,epsxs[N],epsys[N])
            # Update reflection matrix
            Reflection = matmul(matmul(self.R,self.G),np.transpose(self.R))
            # Update the electrical field downward propagation to the scattering interface
            Prop_down = matmul(Prop_down,matmul(matmul(self.R,self.T),np.transpose(self.R)))
            # Update upward propagation
            Prop_up = matmul(Prop_up,matmul(matmul(self.R,self.T),np.transpose(self.R)))

        if D is None:
            self.S = matmul(matmul(Prop_down,Reflection),Prop_up)
        else:
            self.attenuation(z)
            self.S = self.D*matmul(matmul(Prop_down,Reflection),Prop_up)
