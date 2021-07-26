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
    Effective Medium model based on Fujita et al. (2006)
    """

    def __init__(self,fc):
        self.c = 3e8                    # free space wave speed
        self.eps0 = 8.85418782e-12      # free space permittivity
        self.mu0 = 1.25663706e-6        # free space permeability
        self.epsr = 3.12                # relative permittivity for ice
        self.fc = fc                    # system center frequency
        self.omega = 2.*np.pi*fc        # angular frequency
        self.lambda0 = self.c/self.fc   # system wavelength in free space
        self.k0 = self.omega/self.c     # system wavenumber

    def delta_epsilon(self,Temp):
        """
        Dielectric Anisotropy
        Temperature based anisotropy constant from
        Fujita et al. (2006) eq. 3
        """
        self.deps = 0.0256 + 3.57e-5*Temp

    def reflection(self,gammax,gammay):
        """
        Reflection (Scattering) Matrix
        Fujita et al. (2006) eq. 8
        """
        self.G = np.array([[gammax,0],[0,gammay]])

    def rotation(self,theta):
        """
        Rotation Matrix
        Fujita et al. (2006) eq. 10
        """
        self.R = np.array([[np.cos(theta),-np.sin(theta)],
                           [np.sin(theta),np.cos(theta)]])

    def freespace(self,z):
        """
        Free space propagation.
        Covers terms in Fujita et al. (2006) eqs. 9 and 12
        """
        self.D = np.exp(1j*self.k0*z)/(4.*np.pi*z)

    def transmission(self,dz,epsx,epsy,eps2x=0.,eps2y=0.):
        """
        Transmission Matrix
        Describes transmission of radio waves through birefringent
        ice, both the phase and amplitude of waves along the principal
        orientations.
        """

        # Electrical conductivity
        sigmax = self.omega*self.eps0*eps2x
        sigmay = self.omega*self.eps0*eps2y

        # Propagation constants (Fujita et al., 2006; eq 7)
        kx = np.sqrt(self.eps0*self.mu0*epsx*self.omega**2.+\
                1j*self.mu0*sigmax*self.omega)
        ky = np.sqrt(self.eps0*self.mu0*epsy*self.omega**2.+\
                1j*self.mu0*sigmay*self.omega)

        # Transmission components (Fujita et al., 2006; eq 6)
        Tx = np.exp(-1j*self.k0*dz+1j*kx*dz)
        Ty = np.exp(-1j*self.k0*dz+1j*ky*dz)

        # Transmission matrix (Fujita et al., 2006; eq 5)
        self.T = np.array([[Tx, 0],
                           [0, Ty]])

    def single_depth_solve(self,z,dz,thetas,epsxs,epsys,D=None):
        """
        Solve at a single depth for all 4 polarizations
        """

        # Number of layers
        N = int(z//dz)
        # If the ice properties are constant fill in placeholder arrays
        if type(thetas) is float:
            thetas = thetas*np.ones(N+1)
        if type(epsxs) is float:
            epsxs = epsxs*np.ones(N+1)
        if type(epsys) is float:
            epsys = epsys*np.ones(N+1)

        # Initiate propagation arrays as identity
        Prop_down = np.eye(2).astype(np.complex)
        Prop_up = np.eye(2).astype(np.complex)

        # For each layer, update the propagation arrays
        for layer in range(N):
            self.rotation(thetas[layer])
            self.transmission(dz,epsxs[layer],epsys[layer])
            # Update the electrical field downward propagation to the scattering interface
            Prop_down = matmul(Prop_down,matmul(matmul(self.R,self.T),np.transpose(self.R)))
            # Update upward propagation
            Prop_up = matmul(Prop_up,matmul(matmul(self.R,self.T),np.transpose(self.R)))

        # Some depth into the final layer
        if z%dz > 0:
            self.rotation(thetas[N])
            self.transmission(z%dz,epsxs[N],epsys[N])
            Prop_down = matmul(Prop_down,matmul(matmul(self.R,self.T),np.transpose(self.R)))
            Prop_up = matmul(Prop_up,matmul(matmul(self.R,self.T),np.transpose(self.R)))

        # Set the reflection matrix based on the angle, theta
        Reflection = matmul(matmul(self.R,self.G),np.transpose(self.R))

        # Return scattering matrix
        if D is None:
            self.S = matmul(matmul(Prop_down,Reflection),Prop_up)
        else:
            # Add the free space propagation term if desired
            self.attenuation(z)
            self.S = self.D**2.*matmul(matmul(Prop_down,Reflection),Prop_up)


    def solve(self,zs,dz,thetas,epsxs,epsys,D=None):
        """
        Solve for a full column return of all 4 polarizations
        """
        self.range = zs
        self.shh = np.empty(len(zs)).astype(complex)
        self.svv = np.empty(len(zs)).astype(complex)
        self.shv = np.empty(len(zs)).astype(complex)
        self.svh = np.empty(len(zs)).astype(complex)
        for i,z in enumerate(zs):
            self.single_depth_solve(z,dz,thetas,epsxs,epsys,D=D)
            self.shh[i] = self.S[0,0]
            self.shv[i] = self.S[0,1]
            self.svh[i] = self.S[1,0]
            self.svv[i] = self.S[1,1]
