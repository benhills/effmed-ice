#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 1 2023

@author: benhills
"""

import numpy as np
from numpy import matmul
from .indicatrices import get_indicatrix


class effective_medium():
    """
    Effective medium model for electromagnetic wave propagation through
    an anisotropic medium (constants are for ice)
    Fujita et al. (2006)
        Radio-wave depolarization and scattering within ice sheets:
        a matrix-based model to link radar and ice-core measurements
        and its application
    Matsuoka et al. (2009)
        Effects of Birefringence Within Ice Sheets on
        Obliquely Propagating Radio Waves
    """

    def __init__(self):
        self.c = 3e8                    # free space wave speed
        self.eps0 = 8.85418782e-12      # free space permittivity
        self.mu0 = 1.25663706e-6        # free space permeability


    def system_setup(self, fc=3e8, psi_w=0., theta_w=0.,
                     antenna_sep=None, H=None, antenna_pol='VV'):
        """
        Specify system parameters including waveform characteristics
        and antenna polarizations

        Parameters
        ----------
        fc:             float,  center frequency of transmitted wave (Hz)
        theta_w:        float,  polar angle of antenna array
        psi_w:          float,  azimuthal angle of antenna array
        antenna_sep:    float,  antenna separation (m)
        H:              float,  ice thickness (m)
        antenna_pol:    str,    antenna polarization [transmit, receive]
        """

        self.fc = fc                    # system center frequency
        self.omega = 2.*np.pi*fc        # angular frequency
        self.lambda0 = self.c/self.fc   # system wavelength in free space
        self.k0 = self.omega/self.c     # system wavenumber

        # antenna azimuthal rotation
        self.psi_w = psi_w
        # propagation angle based on antenna separation
        if theta_w != 0. and antenna_sep is not None:
            raise ValueError('Cannot set both theta_w and antenna_sep, only set one.')
        elif antenna_sep is not None:
            if H is None:
                raise ValueError('If using antenna_sep must also give ice thickness, H.')
            self.theta_w = np.arccos(H/antenna_sep)
        else:
            self.theta_w = theta_w

        # transmit polarization
        if antenna_pol[0] == 'V':
            self.Wt = np.array([1, 0])
        elif antenna_pol[0] == 'H':
            self.Wt = np.array([0, 1])
        else:
            raise('Only V and H polarizations are accepted.')

        # receive polarization
        if antenna_pol[1] == 'V':
            self.Wr = np.array([1, 0])
        elif antenna_pol[1] == 'H':
            self.Wr = np.array([0, 1])
        else:
            raise('Only V and H polarizations are accepted, prescribe angle to rotate.')


    def ice_properties(self, fabric='vertical-girdle', T=None, epsr=3.12, epsc=0.,
                       chi=[.5, 0., .5], theta=0., psi=0.):
        """
        Set the ice properties including permittivity
        and crystal orientation fabric

        Parameters
        ----------
        fabric:     str,    qualitative fabric 'type' for indicatrix selection
        T:          float,  ice temperature
        epsr:       float,  relative permittivity
        epsr2:       float,  complex relative permittivity
        chi:        array,  c-axes distribution (eigenvalues)
        theta:      float,  polar angle of vertical eigenvector (chi[2])
        psi:        float,  azimuthal angle of vertical eigenvalue (chi[2])
        """

        self.fabric = fabric

        # Save input variables to the model class
        self.mr_perp = np.sqrt(epsr)    # relative permittivity of ice for perpendicular polarization
        self.mc_perp = np.sqrt(epsc)    # complex permittivity of ice for perpendicular polarization
        self.chi = chi                  # eigenvalues of the c-axes distribution

        if T is not None:
            # Temperature based anisotropy constant from
            # Fujita et al. (2006) eq. 3
            self.depsr = 0.0256 + 3.57e-5 * T
            self.depsc = 0.0
        else:
            # for ice at approximately -35 C
            self.depsr = 0.034
            self.depsc = 0.0

        # set the parallel permittivity
        self.mr_par = np.sqrt(self.mr_perp**2. + self.depsr)
        self.mc_par = np.sqrt(self.mc_perp**2. + self.depsc)

        # use external functions to get the indicatrix
        get_indicatrix(self, fabric, theta, psi)


    def reflection(self,gammax=1., gammay=1.):
        """
        Reflection (Scattering) Matrix
        Fujita et al. (2006) eq. 8

        Parameters
        ----------
        gammax:     float,  reflectivity in x-dir
        gammay:     float,  reflectivity in y-dir
        """
        self.G = np.array([[gammax,0],[0,gammay]])


    def rotation(self, phi):
        """
        Rotation Matrix
        Fujita et al. (2006) eq. 10

        Parameters
        ----------
        theta:      float,  angle
        """

        self.R = np.array([[np.cos(phi), -np.sin(phi)],
                           [np.sin(phi), np.cos(phi)]])


    def freespace(self,z):
        """
        Free space propagation.
        Covers terms in Fujita et al. (2006) eqs. 9 and 12

        Parameters
        ----------
        z:          float,  vertical distance
        """

        self.D = np.exp(1j*self.k0*z)/(4.*np.pi*z)


    def transmission(self,dz):
        """
        Transmission Matrix
        Describes transmission of radio waves through birefringent
        ice, both the phase and amplitude of waves along the principal
        orientations.

        Parameters
        ----------
        dz:      float,  layer thickness (m)
        #TODO from matsuoka model??
        #l = d/np.cos(theta_w)
        """

        # Propagation distance
        dl = dz*np.cos(self.theta_w)

        # Transmission components (Fujita et al., 2006; eq 6)
        T_1 = np.exp(-1j*self.k0*dz+1j*self.k_1*dl)
        T_2 = np.exp(-1j*self.k0*dz+1j*self.k_2*dl)

        # Transmission matrix (Fujita et al., 2006; eq 5)
        self.T = np.array([[T_1, 0],
                           [0, T_2]])


    def single_depth_solve(self,z,dz,phis,chis,gamma=[1., 1.],D=None):
        """
        Solve at a single depth for all 4 polarizations

        Parameters
        ----------
        gamma:     float,  reflectivity in x-dir

        """

        # Number of layers
        N = int(z//dz)
        # If the ice properties are constant fill in placeholder arrays
        if type(phis) is float and np.shape(chis) == (3,):
            phi = phis
            uniform = True
        else:
            uniform = False

        # Initiate propagation arrays as identity
        Prop_down = np.eye(2).astype(np.complex)
        Prop_up = np.eye(2).astype(np.complex)

        # For each layer, update the propagation arrays
        for layer in range(N):
            if uniform:
               pass
            else:
                phi = phis[layer]
                self.ice_properties(fabric=self.fabric, chi=chis[layer])
            self.rotation(phi)
            self.transmission(dz)
            # Update the electrical field downward propagation to the scattering interface
            Prop_down = matmul(Prop_down, matmul(matmul(self.R, self.T), np.transpose(self.R)))
            # Update upward propagation
            Prop_up = matmul(Prop_up, matmul(matmul(self.R, self.T), np.transpose(self.R)))

        # Some depth into the final layer
        if z % dz > 0:
            if uniform:
                pass
            else:
                phi = phis[layer]
                self.ice_properties(fabric=self.fabric, chi=chis[layer])
            self.rotation(phi)
            self.transmission(z%dz)
            Prop_down = matmul(Prop_down,matmul(matmul(self.R,self.T),np.transpose(self.R)))
            Prop_up = matmul(Prop_up,matmul(matmul(self.R,self.T),np.transpose(self.R)))

        # Set the reflection matrix
        self.reflection(gamma[0], gamma[1])
        Reflection = matmul(matmul(self.R, self.G), np.transpose(self.R))

        # Return scattering matrix
        if D is None:
            self.S = matmul(matmul(Prop_down,Reflection),Prop_up)
        else:
            # Add the free space propagation term if desired
            self.attenuation(z)
            self.S = self.D**2.*matmul(matmul(Prop_down,Reflection),Prop_up)


    def solve(self, zs, dz, phis, chis, gammas=None, D=None):
        """
        Solve for a full column return of all 4 polarizations

        Parameters
        ----------
        gammas:     float,  reflectivity in x-dir
        gammay:     float,  reflectivity in y-dir

        """

        self.range = zs
        self.shh = np.empty(len(zs)).astype(complex)
        self.svv = np.empty(len(zs)).astype(complex)
        self.shv = np.empty(len(zs)).astype(complex)
        self.svh = np.empty(len(zs)).astype(complex)
        for i, z in enumerate(zs):
            if gammas is None:
                gamma = [1., 1.]
            elif np.shape(gammas) == (2,):
                gamma = gammas
            else:
                gamma = gammas[i]
            self.single_depth_solve(z, dz, phis, chis, gamma, D=D)
            # TODO: Is this right?? seems wrong
            self.shh[i] = self.S[0, 0]
            self.shv[i] = self.S[0, 1]
            self.svh[i] = self.S[1, 0]
            self.svv[i] = self.S[1, 1]
