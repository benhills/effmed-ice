#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 1 2023

@author: benhills
"""

import numpy as np
from numpy import matmul
from .indicatrices import get_prop_const


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
                     antenna_sep=None, H=None):
        """
        Specify system parameters including waveform characteristics
        and antenna polarizations

        Parameters
        ----------
        fc:             float,  center frequency of transmitted wave (Hz)
        psi_w:          float,  azimuthal angle of antenna array
        theta_w:        float,  polar angle of antenna array
        antenna_sep:    float,  antenna separation (m)
        H:              float,  ice thickness (m)
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
            self.theta_w = np.arctan(antenna_sep/H)
        else:
            self.theta_w = theta_w


    def ice_properties(self, idctx='vertical-girdle', T=None, epsr=3.12, epsi=0.,
                       theta=0., psi= 0., chi=[.5, 0., .5], prop_up=False):
        """
        Set the ice properties including permittivity
        and crystal orientation fabric

        Parameters
        ----------
        idctx:      str,    qualitative fabric 'type' for indicatrix selection
        T:          float,  ice temperature
        epsr:       float,  real relative permittivity
        epsc:       float,  imaginary relative permittivity
        theta:      float,  polar angle of vertical eigenvector (chi[2])
        psi:        float,  azimuthal angle of vertical eigenvalue or girdle
        chi:        array,  c-axes distribution (eigenvalues)
        prop_up:    bool,   propagate up? changes the indicatrix output
        """

        # Save input variables to the model class
        self.mr_perp = np.sqrt(epsr)    # real relative permittivity of ice for perpendicular polarization
        self.mi_perp = np.sqrt(epsi)    # imaginary relative permittivity of ice for perpendicular polarization
        self.chi = chi                  # eigenvalues of the c-axes distribution

        if T is not None:
            # Temperature based anisotropy constant from
            # Fujita et al. (2006) eq. 3
            self.depsr = 0.0256 + 3.57e-5 * T
            self.depsi = 0j #TODO: get the imaginary component
        else:
            # for ice at approximately -35 C
            self.depsr = 0.034
            self.depsi = 0j #TODO: get the imaginary component

        # set the parallel permittivity
        self.mr_par = np.sqrt(self.mr_perp**2. + self.depsr)
        self.mi_par = np.sqrt(self.mi_perp**2. + self.depsi)

        # use external functions to get the indicatrix
        get_prop_const(self, idctx, theta, psi, prop_up)


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


    def rotation(self, psi):
        """
        Rotation Matrix
        Fujita et al. (2006) eq. 10

        Parameters
        ----------
        psi:    float,  angle
        """

        self.R = np.array([[np.cos(psi), -np.sin(psi)],
                           [np.sin(psi), np.cos(psi)]])


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
        """

        # Propagation distance
        dl = dz/np.cos(self.theta_w)

        # Transmission components (Fujita et al., 2006; eq 6)
        T_1 = np.exp(-1j*self.k0*dl+1j*self.k[0]*dl)
        T_2 = np.exp(-1j*self.k0*dl+1j*self.k[1]*dl)

        # Transmission matrix (Fujita et al., 2006; eq 5)
        self.T = np.array([[T_1, 0],
                           [0, T_2]])


    def single_depth_solve(self,z,dzs,thetas,psis,chis,gamma=[1., 1.],psi_gamma=None,
                           idctx='vertical-girdle', free_space=False):
        """
        Solve at a single depth for all 4 polarizations

        Parameters
        ----------
        z:          float,      total propagation depth in the vertical
        dzs:        Nx1-array,  layer thicknesses
        thetas:     Nx1-array,  polar angles for each layer
        psis:       Nx1-array,  azimuthal angles for each layer
        chis:       Nx3-array,  eigenvalues of the crystal orientation fabric for each layer
        gamma:      array,      reflectivity in x and y directions
        idctx:      str,        qualitative fabric 'type' for indicatrix selection
        free_space: bool,       include free space transmission losses or not
        """

        # If the ice properties are constant fill in placeholder arrays
        if type(thetas) is float and type(psis) is float and np.shape(chis) == (3,):
            dzs = np.array([dzs, dzs])
            psis = np.array([psis, psis])
            thetas = np.array([thetas, thetas])
            chis = np.array([chis, chis])

        # Initiate propagation arrays as identity
        Prop_down = np.eye(2).astype(complex)
        Prop_up = np.eye(2).astype(complex)

        # Propagate downward, updating the rotated transmission matrix through each layer
        layer_n, z_prop = 0, 0
        while (z_prop+dzs[-1]) < z:
            self.ice_properties(idctx=idctx, chi=chis[layer_n], psi=psis[layer_n], theta=thetas[layer_n])
            self.rotation(self.psi_)
            self.transmission(dzs[layer_n])
            # Update the electrical field downward propagation to the scattering interface
            Prop_down = matmul(Prop_down, matmul(matmul(self.R, self.T), np.transpose(self.R)))
            # Update loop constants
            z_prop += dzs[layer_n]
            layer_n += 1

        # Some depth into the final layer
        self.ice_properties(idctx=idctx, chi=chis[layer_n], psi=psis[layer_n], theta=thetas[layer_n])
        self.rotation(self.psi_)
        self.transmission(z-z_prop)
        Prop_down = matmul(Prop_down,matmul(matmul(self.R,self.T),np.transpose(self.R)))


        # Set the reflection matrix
        # ------------------------------------------- #
        if psi_gamma is not None:
            self.rotation(psi_gamma)
        self.reflection(gamma[0], gamma[1])
        Reflection = matmul(matmul(self.R, self.G), np.transpose(self.R))


        ### Propagate upward, updating scattering matrix through each layer
        # ------------------------------------------- #
        self.ice_properties(idctx=idctx, chi=chis[layer_n], psi=psis[layer_n], theta=thetas[layer_n], prop_up=True)
        self.rotation(self.psi__)
        self.transmission(z-z_prop)
        Prop_up = matmul(Prop_up,matmul(matmul(self.R,self.T),np.transpose(self.R)))

        layer_n -= 1
        while (z_prop-dzs[0]) >= 0:
            self.ice_properties(idctx=idctx, chi=chis[layer_n], psi=psis[layer_n], theta=thetas[layer_n], prop_up=True)
            self.rotation(self.psi__)
            self.transmission(dzs[layer_n])
            # Update the electrical field upward propagation to the scattering interface
            Prop_up = matmul(Prop_up, matmul(matmul(self.R, self.T), np.transpose(self.R)))
            # Update loop constants
            z_prop -= dzs[layer_n]
            layer_n -= 1


        # Return scattering matrix
        # ------------------------------------------- #
        if free_space is False:
            self.S = matmul(matmul(Prop_up,Reflection),Prop_down)
        else:
            # Add the free space propagation term if desired
            self.freespace(z)
            self.S = self.D**2.*matmul(matmul(Prop_up,Reflection),Prop_down)


    def solve(self, zs, dzs, thetas, psis, chis,
              idctx='vertical-girdle', gammas=None, psi_gammas=None, free_space=False):
        """
        Solve for a full column return of all 4 polarizations

        Parameters
        ----------
        zs:         array,      monotonically increasing depth array (from top to bottom, z)
        dzs:        Nx1-array,  layer thicknesses
        thetas:     Nx1-array,  polar angles for each layer
        psis:       Nx1-array,  azimuthal angles for each layer
        chis:       Nx3-array,  eigenvalues of the crystal orientation fabric for each layer
        gammas:     array,      reflectivity in x and y directions for each layer (or same for all layers)
        psi_gammas: array,      azimuth of scattering interface for each layer
        idctx:      str,        qualitative fabric 'type' for indicatrix selection
        free_space: bool,       include free space transmission losses or not
        """

        # Pre-assign output arrays
        self.range = zs
        self.shh = np.empty(len(zs)).astype(complex)
        self.svv = np.empty(len(zs)).astype(complex)
        self.shv = np.empty(len(zs)).astype(complex)
        for i, z in enumerate(zs):
            # Get the reflectivity for this depth
            if gammas is None:
                gamma = [1., 1.]
            elif np.shape(gammas) == (2,):
                gamma = gammas
            else:
                gamma = gammas[i]
            # Get the azimuth of the scattering interface
            if psi_gammas is None:
                psi_gamma = None
            elif not hasattr(psi_gammas,'len'):
                psi_gamma = psi_gammas
            else:
                psi_gamma = psi_gammas[i]

            # Do the wave propagation solve at this depth
            self.single_depth_solve(z, dzs, thetas, psis, chis, gamma,
                                    psi_gamma=psi_gamma, idctx=idctx, free_space=free_space)

            # Assign results for each polarization to their respective array
            self.shh[i] = self.S[0, 0]
            self.svv[i] = self.S[1, 1]
            self.shv[i] = self.S[0, 1]
        self.svh = self.shv.copy()
