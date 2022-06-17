#!/usr/bin/env python

"""Utilities for setting up NICER and XMM Newtwon analysis.

FIXME: these settings are specific to the STU model with J0740
"""

import numpy as np
import math
import six as _six

from scipy.interpolate import Akima1DInterpolator

import xpsi
from xpsi import Parameter, make_verbose
from xpsi.likelihoods.default_background_marginalisation import eval_marginal_likelihood
from xpsi.likelihoods.default_background_marginalisation import precomputation



#
#   Following copied from the STU directory in:
#   A_NICER_VIEW_OF_PSR_J0740+6620.tar.gz
#   which was downloaded from https://zenodo.org/record/5735003
#

class CustomInstrument(xpsi.Instrument):
    """ NICER XTI, and XMM pn, MOS1, and MOS2 instruments. """
    def construct_matrix(self):
        """ Implement response matrix parameterisation. """
        matrix = self['alpha'] * self.matrix # * self['absolute_alpha']
        matrix[matrix < 0.0] = 0.0

        return matrix

    def __call__(self, signal, *args):
        """ Overwrite. """

        matrix = self.construct_matrix()

        self._cached_signal = np.dot(matrix, signal)

        return self._cached_signal

    @classmethod
    @make_verbose('Loading NICER XTI response matrix',
                  'Response matrix loaded')
    def NICER_XTI(cls,
                  bounds, values,
                  ARF, RMF,
                  max_input,
                  max_channel,
                  min_input=0,
                  min_channel=0,
                  channel_edges=None,
                  **kwargs):
        """ Load NICER XTI instrument response matrix. """
        ARF = np.loadtxt(ARF, dtype=np.double, skiprows=3)
        RMF = np.loadtxt(RMF, dtype=np.double, skiprows=3, usecols=-1)

        if channel_edges:
            channel_edges = np.loadtxt(channel_edges, dtype=np.double, skiprows=3)

        matrix = np.zeros((1501,3451))

        for i in range(3451):
            matrix[:,i] = RMF[i*1501:(i+1)*1501]

        if min_input != 0:
            min_input = int(min_input)

        max_input = int(max_input)

        edges = np.zeros(max_input - min_input + 1, dtype=np.double)

        edges[0] = ARF[min_input,1]; edges[1:] = ARF[min_input:max_input,2]

        RSP = np.zeros((max_channel - min_channel,
                        max_input - min_input), dtype=np.double)

        for i in range(RSP.shape[0]):
            RSP[i,:] = matrix[i+min_channel, min_input:max_input] * ARF[min_input:max_input,3] * 51.0/52.0

        channels = np.arange(min_channel, max_channel)

        alpha = Parameter('alpha',
                          strict_bounds = (0.1,1.9),
                          bounds = bounds.get('alpha', None),
                          doc='NICER XTI energy-independent scaling factor',
                          symbol = r'$\alpha_{\rm XTI}$',
                          value = values.get('alpha', None))

        return cls(RSP, edges, channels, channel_edges[min_channel:max_channel+1,1],
                   alpha, **kwargs)

    @classmethod
    @make_verbose('Loading XMM-pn response matrix',
                  'Response matrix loaded')
    def XMM_PN(cls,
               bounds, values,
               ARF, RMF,
               max_input,
               max_channel,
               min_input=0,
               min_channel=0,
               channel_edges=None,
               **kwargs):
        """ Load XMM-pn instrument response matrix. """
        ARF = np.loadtxt(ARF, dtype=np.double, skiprows=3)
        RMF = np.loadtxt(RMF, dtype=np.double, skiprows=3, usecols=-1)

        if channel_edges:
            channel_edges = np.loadtxt(channel_edges, dtype=np.double, skiprows=3)

        if min_input != 0:
            min_input = int(min_input)

        max_input = int(max_input)

        matrix = np.zeros((4096, max_input))

        for i in range(max_input - min_input):
            matrix[:,i] = RMF[i*4096:(i+1)*4096]

        edges = np.zeros(max_input - min_input + 1, dtype=np.double)

        edges[0] = ARF[min_input,1]; edges[1:] = ARF[min_input:max_input,2]

        RSP = np.zeros((max_channel - min_channel,
                        max_input - min_input), dtype=np.double)

        for i in range(RSP.shape[0]):
            RSP[i,:] = matrix[i+min_channel,min_input:max_input] * ARF[min_input:max_input,3]

        channels = np.arange(min_channel, max_channel)

        alpha = Parameter('alpha',
                          strict_bounds = (0.1,1.9),
                          bounds = bounds.get('alpha', None),
                          doc='XMM-pn energy-independent scaling factor',
                          symbol = r'$\alpha_{\rm pn}$',
                          value = values.get('alpha', None))

        return cls(RSP, edges, channels,
                   channel_edges[min_channel:max_channel+1,1],
                   alpha, **kwargs)

    @classmethod
    @make_verbose('Loading XMM-MOS1 response matrix',
                  'Response matrix loaded')
    def XMM_MOS1(cls,
                 bounds, values,
                 ARF, RMF,
                 max_input,
                 max_channel,
                 min_input=0,
                 min_channel=0,
                 channel_edges = None,
                 **kwargs):
        """ Load XMM-MOS1 instrument response matrix. """
        ARF = np.loadtxt(ARF, dtype=np.double, skiprows=9)
        RMF = np.loadtxt(RMF, dtype=np.double, skiprows=9, usecols=-1)

        if channel_edges:
            channel_edges = np.loadtxt(channel_edges, dtype=np.double, skiprows=3)

        if min_input != 0:
            min_input = int(min_input)

        max_input = int(max_input)

        matrix = np.zeros((800, max_input))

        for i in range(matrix.shape[1]):
            matrix[:,i] = RMF[i*800:(i+1)*800]

        edges = np.zeros(max_input - min_input + 1, dtype=np.double)

        edges[0] = ARF[min_input,1]; edges[1:] = ARF[min_input:max_input,2]

        RSP = np.zeros((max_channel - min_channel,
                        max_input - min_input), dtype=np.double)

        for i in range(RSP.shape[0]):
            RSP[i,:] = matrix[i+min_channel,min_input:max_input] * ARF[min_input:max_input,3]

        channels = np.arange(min_channel, max_channel)

        alpha = Parameter('alpha',
                          strict_bounds = (0.1,1.9),
                          bounds = bounds.get('alpha', None),
                          doc='XMM-MOS1 energy-independent scaling factor',
                          symbol = r'$\alpha_{\rm MOS1}$',
                          value = values.get('alpha', None))

        return cls(RSP, edges, channels,
                   channel_edges[min_channel:max_channel+1,1],
                   alpha, **kwargs)

    @classmethod
    @make_verbose('Loading XMM-MOS2 response matrix',
                  'Response matrix loaded')
    def XMM_MOS2(cls,
                 bounds, values,
                 ARF, RMF,
                 max_input,
                 max_channel,
                 min_input=0,
                 min_channel=0,
                 channel_edges = None,
                 **kwargs):
        """ Load XMM-MOS2 instrument response matrix. """
        ARF = np.loadtxt(ARF, dtype=np.double, skiprows=9)
        RMF = np.loadtxt(RMF, dtype=np.double, skiprows=9, usecols=-1)

        if channel_edges:
            channel_edges = np.loadtxt(channel_edges, dtype=np.double, skiprows=3)

        if min_input != 0:
            min_input = int(min_input)

        max_input = int(max_input)

        matrix = np.zeros((800, max_input))

        for i in range(matrix.shape[1]):
            matrix[:,i] = RMF[i*800:(i+1)*800]

        edges = np.zeros(max_input - min_input + 1, dtype=np.double)

        edges[0] = ARF[min_input,1]; edges[1:] = ARF[min_input:max_input,2]

        RSP = np.zeros((max_channel - min_channel,
                        max_input - min_input), dtype=np.double)

        for i in range(RSP.shape[0]):
            RSP[i,:] = matrix[i+min_channel,min_input:max_input] * ARF[min_input:max_input,3]

        channels = np.arange(min_channel, max_channel)

        alpha = Parameter('alpha',
                          strict_bounds = (0.1,1.9),
                          bounds = bounds.get('alpha', None),
                          doc='XMM-MOS2 energy-independent scaling factor',
                          symbol = r'$\alpha_{\rm MOS2}$',
                          value = values.get('alpha', None))

        return cls(RSP, edges, channels,
                   channel_edges[min_channel:max_channel+1,1],
                   alpha, **kwargs)


class CustomSignal(xpsi.Signal):
    """ A custom calculation of the logarithm of the NICER likelihood.

    We extend the :class:`xpsi.Signal.Signal` class to make it callable.

    We overwrite the body of the __call__ method. The docstring for the
    abstract method is copied.

    """

    def __init__(self, workspace_intervals = 1000, epsabs = 0, epsrel = 1.0e-8,
                 epsilon = 1.0e-3, sigmas = 10.0, support = None, *args, **kwargs):
        """ Perform precomputation. """

        super(CustomSignal, self).__init__(*args, **kwargs)

        try:
            self._precomp = precomputation(self._data.counts.astype(np.int32))
        except AttributeError:
            print('No data... can synthesise data but cannot evaluate a '
                  'likelihood function.')
        else:
            self._workspace_intervals = workspace_intervals
            self._epsabs = epsabs
            self._epsrel = epsrel
            self._epsilon = epsilon
            self._sigmas = sigmas

            if support is not None:
                self._support = support
            else:
                self._support = -1.0 * np.ones((self._data.counts.shape[0],2))
                self._support[:,0] = 0.0

    @property
    def support(self):
        return self._support

    @support.setter
    def support(self, obj):
        self._support = obj

    def __call__(self, *args, **kwargs):
        self.loglikelihood, self.expected_counts, self.background_signal, self.background_signal_given_support = \
                eval_marginal_likelihood(self._data.exposure_time,
                                          self._data.phases,
                                          self._data.counts,
                                          self._signals,
                                          self._phases,
                                          self._shifts,
                                          self._precomp,
                                          self._support,
                                          self._workspace_intervals,
                                          self._epsabs,
                                          self._epsrel,
                                          self._epsilon,
                                          self._sigmas,
                                          kwargs.get('llzero'))
 

class CustomInterstellar(xpsi.Interstellar):
    """ Apply interstellar attenuation. """

    def __init__(self, energies, attenuation, bounds, values = {}):

        assert len(energies) == len(attenuation), 'Array length mismatch.'

        self._lkp_energies = energies # for lookup
        self._lkp_attenuation = attenuation # for lookup

        N_H = Parameter('column_density',
                        strict_bounds = (0.0,10.0),
                        bounds = bounds.get('column_density', None),
                        doc = 'Units of 10^20 cm^-2.',
                        symbol = r'$N_{\rm H}$',
                        value = values.get('column_density', None))

        self._interpolator = Akima1DInterpolator(self._lkp_energies,
                                                 self._lkp_attenuation)
        self._interpolator.extrapolate = True

        super(CustomInterstellar, self).__init__(N_H)

    def attenuation(self, energies):
        """ Interpolate the attenuation coefficients.

        Useful for post-processing.

        """
        return self._interpolate(energies)**(self['column_density']/0.4)

    def _interpolate(self, energies):
        """ Helper. """
        _att = self._interpolator(energies)
        _att[_att < 0.0] = 0.0
        return _att

    @classmethod
    def from_SWG(cls, path, **kwargs):
        """ Load attenuation file from the NICER SWG. """

        temp = np.loadtxt(path, dtype=np.double)

        energies = temp[0:351,0]

        attenuation = temp[0:351,2]

        return cls(energies, attenuation, **kwargs)


class CustomPhotosphere(xpsi.Photosphere):
    """ A photosphere extension to preload the numerical atmosphere NSX.

    Fully-ionized hydrogen, v200802 (W.C.G. Ho).

    """

    @xpsi.Photosphere.hot_atmosphere.setter
    def hot_atmosphere(self, path):
        NSX = np.loadtxt(path, dtype=np.double)
        logT = np.zeros(35)
        logg = np.zeros(14)
        mu = np.zeros(67)
        logE = np.zeros(166)

        reorder_buf = np.zeros((35,14,67,166))

        index = 0
        for i in range(reorder_buf.shape[0]):
            for j in range(reorder_buf.shape[1]):
                for k in range(reorder_buf.shape[3]):
                   for l in range(reorder_buf.shape[2]):
                        logT[i] = NSX[index,3]
                        logg[j] = NSX[index,4]
                        logE[k] = NSX[index,0]
                        mu[reorder_buf.shape[2] - l - 1] = NSX[index,1]
                        reorder_buf[i,j,reorder_buf.shape[2] - l - 1,k] = 10.0**(NSX[index,2])
                        index += 1

        buf = np.zeros(np.prod(reorder_buf.shape))

        bufdex = 0
        for i in range(reorder_buf.shape[0]):
            for j in range(reorder_buf.shape[1]):
                for k in range(reorder_buf.shape[2]):
                   for l in range(reorder_buf.shape[3]):
                        buf[bufdex] = reorder_buf[i,j,k,l]; bufdex += 1

        self._hot_atmosphere = (logT, logg, mu, logE, buf)

    @property
    def global_variables(self):
        """ This method is needed if we also want to invoke the image-plane signal simulator.

        The extension module compiled is surface_radiation_field/archive/local_variables/two_spots.pyx,
        which replaces the contents of surface_radiation_field/local_variables.pyx.

        """
        return np.array([self['p__super_colatitude'],
                          self['p__phase_shift'] * 2.0 * math.pi,
                          self['p__super_radius'],
                          self['p__super_temperature'],
                          self['s__super_colatitude'],
                          (self['s__phase_shift'] + 0.5) * 2.0 * math.pi,
                          self['s__super_radius'],
                          self['s__super_temperature']])
