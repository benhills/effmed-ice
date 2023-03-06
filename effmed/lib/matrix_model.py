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
            self.theta_w = np.arccos(H/antenna_sep)
        else:
            self.theta_w = theta_w


    def ice_properties(self, idctx='none', T=None, epsr=3.12, epsc=0.,
                       theta=0., psi= 0., chi=[.5, 0., .5], prop_up=False):
        """
        Set the ice properties including permittivity
        and crystal orientation fabric

        Parameters
        ----------
        idctx:     str,    qualitative fabric 'type' for indicatrix selection
        T:              float,  ice temperature
        epsr:           float,  relative permittivity
        epsc:           float,  complex relative permittivity
        theta:          float,  polar angle of vertical eigenvector (chi[2])
        psi:            float,  azimuthal angle of vertical eigenvalue (chi[2])
        chi:            array,  c-axes distribution (eigenvalues)
        prop_up:        bool,   propagate up? changes the indicatrix output
        """

        # Save input variables to the model class
        self.mr_perp = np.sqrt(epsr)    # relative permittivity of ice for perpendicular polarization
        self.mc_perp = np.sqrt(epsc)    # complex permittivity of ice for perpendicular polarization
        self.chi = chi                  # eigenvalues of the c-axes distribution

        if T is not None:
            # Temperature based anisotropy constant from
            # Fujita et al. (2006) eq. 3
            self.depsr = 0.0256 + 3.57e-5 * T
            self.depsc = 0.0    #TODO: set this
        else:
            # for ice at approximately -35 C
            self.depsr = 0.034
            self.depsc = 0.0    #TODO: set this

        # set the parallel permittivity
        self.mr_par = np.sqrt(self.mr_perp**2. + self.depsr)
        self.mc_par = np.sqrt(self.mc_perp**2. + self.depsc)

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
        theta:      float,  angle
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
        #TODO from matsuoka model??
        #l = d/np.cos(theta_w)
        """

        # Propagation distance
        dl = dz*np.cos(self.theta_w)

        # Transmission components (Fujita et al., 2006; eq 6)
        T_1 = np.exp(-1j*self.k0*dl+1j*self.k_1*dl)
        T_2 = np.exp(-1j*self.k0*dl+1j*self.k_2*dl)

        # Transmission matrix (Fujita et al., 2006; eq 5)
        self.T = np.array([[T_1, 0],
                           [0, T_2]])


    def single_depth_solve(self,z,dzs,thetas,psis,chis,gamma=[1., 1.],
                           idctx='none', D=None, verbose=False):
        """
        Solve at a single depth for all 4 polarizations

        Parameters
        ----------
        z:     float,  reflectivity in x-dir
        dzs:     float,  reflectivity in x-dir
        thetas:     float,  reflectivity in x-dir
        psis:     float,  reflectivity in x-dir
        chis:     float,  reflectivity in x-dir
        gamma:     float,  reflectivity in x-dir
        D:     float,  reflectivity in x-dir
        """

        # If the ice properties are constant fill in placeholder arrays
        if type(thetas) is float and type(psis) is float and np.shape(chis) == (3,):
            dzs = np.array([dzs])
            psis = np.array([psis])
            thetas = np.array([thetas])
            chis = np.array([chis])
            self.ice_properties(idctx=idctx, theta=thetas[0], psi=psis[0], chi=chis[0])
            uniform = True
        else:
            uniform = False

        # Initiate propagation arrays as identity
        Prop_down = np.eye(2).astype(complex)
        Prop_up = np.eye(2).astype(complex)

        # Propagate downward, updating scattering matrix through each layer
        layer_n, z_prop = 0, 0
        while (z_prop+dzs[-1]) < z:
            if uniform:
                pass
            else:
                self.ice_properties(idctx=idctx, chi=chis[layer_n], psi=psis[layer_n], theta=thetas[layer_n])
            self.rotation(self.psi_)
            self.transmission(dzs[layer_n])
            # Update the electrical field downward propagation to the scattering interface
            Prop_down = matmul(Prop_down, matmul(matmul(self.R, self.T), np.transpose(self.R)))

            z_prop += dzs[layer_n]

            if verbose:
                print('prop down layer:',layer_n)
                print('z:', dzs[layer_n])
                print('Chi:', (self.m_1,self.m_2))
                print('Psi:', self.psi_)
                print('Theta:', self.theta_)

            layer_n += 1


        # Some depth into the final layer
        if z_prop < z:
            if uniform:
                pass
            else:
                self.ice_properties(idctx=idctx, chi=chis[layer_n], psi=psis[layer_n], theta=thetas[layer_n])
            self.rotation(self.psi_)
            self.transmission(z-z_prop)
            Prop_down = matmul(Prop_down,matmul(matmul(self.R,self.T),np.transpose(self.R)))

            # Set the reflection matrix
            self.reflection(gamma[0], gamma[1])
            Reflection = matmul(matmul(self.R, self.G), np.transpose(self.R))

            if verbose:
                print('Reflection with Gamma:',gamma)


            if uniform:
                self.ice_properties(idctx=idctx, chi=chis[0], psi=psis[0], theta=thetas[0], prop_up=True)
                pass
            else:
                self.ice_properties(idctx=idctx, chi=chis[layer_n], psi=psis[layer_n], theta=thetas[layer_n], prop_up=True)
            self.rotation(self.psi__)
            self.transmission(z-z_prop)
            Prop_up = matmul(Prop_up,matmul(matmul(self.R,self.T),np.transpose(self.R)))

            if verbose:
                print('prop down/up layer:',layer_n)
                print('z:',z-z_prop)
                print('Chi:', (self.m_1,self.m_2))
                print('Psi:', self.psi_)
                print('Theta:', self.theta_)



        # Propagate upward, updating scattering matrix through each layer
        layer_n -= 1
        while (z_prop-dzs[0]) >= 0:
            if uniform:
                pass
            else:
                self.ice_properties(idctx=idctx, chi=chis[layer_n], psi=psis[layer_n], theta=thetas[layer_n], prop_up=True)
            self.rotation(self.psi__)
            self.transmission(dzs[layer_n])
            # Update the electrical field upward propagation to the scattering interface
            Prop_up = matmul(Prop_up, matmul(matmul(self.R, self.T), np.transpose(self.R)))

            z_prop -= dzs[layer_n]

            if verbose:
                print('prop up layer:',layer_n)
                print('z:', dzs[layer_n])
                print('Chi:', (self.m_1,self.m_2))
                print('Psi:', self.psi_)
                print('Theta:', self.theta_)

            layer_n -= 1

        # Return scattering matrix
        if D is None:
            self.S = matmul(matmul(Prop_up,Reflection),Prop_down)
        else:
            # Add the free space propagation term if desired
            self.attenuation(z)
            self.S = self.D**2.*matmul(matmul(Prop_up,Reflection),Prop_down)


    def solve(self, zs, dzs, thetas, psis, chis, idctx='none', gammas=None, D=None):
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
            self.single_depth_solve(z, dzs, thetas, psis, chis, gamma, idctx=idctx, D=D)
            self.shh[i] = self.S[0, 0]
            self.shv[i] = self.S[0, 1]
            self.svh[i] = self.S[1, 0]
            self.svv[i] = self.S[1, 1]
