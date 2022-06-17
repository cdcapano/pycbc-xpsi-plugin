#!/usr/bin/env python

import numpy
import shlex
import xpsi
from xpsi.Parameter import Derive
from xpsi.global_imports import gravradius

from .utils import (CustomInstrument, CustomSignal, CustomInterstellar,
                    CustomPhotosphere)

from pycbc.workflow import WorkflowConfigParser
from pycbc.inference.models import BaseModel



class XPSIModel(BaseModel):
    """Model wrapper around XPSI likelihood function."""
    name = 'xpsi'

    # we need to alias some of the parameter names to be compliant with
    # pycbc config file sections
    _param_aliases = {
        'XTI__alpha': 'nicer_alpha',
        'PN__alpha': 'xmm_alpha'
        }

    def __init__(self, variable_params, star, signals, num_energies, **kwargs):
        super().__init__(variable_params, **kwargs)
        # set up the xpsi likelihood
        self._xpsi_likelihood = xpsi.Likelihood(star=star, signals=signals,
                                                num_energies=num_energies,
                                                externally_updated=True)
        # store a dictionary of param aliases
        self.param_aliases = {p: p for p in self._xpsi_likelihood.names}
        self.param_aliases.update(self._param_aliases)

    @property
    def star(self):
        return self._xpsi_likelihood.star
        
    def _loglikelihood(self):
        # map the current parameters to the ordered list
        params = self.current_params
        # update the underlying likelihood
        for p in self._xpsi_likelihood.names:
            self._xpsi_likelihood[p] = params[self.param_aliases[p]]
        # check addtional constraints
        if not self.apply_additional_constraints():
            return -numpy.inf
        logl = self._xpsi_likelihood()
        # FIXME: for some reason, this occasionally returns arrays of len 1
        # if so, force to float
        if isinstance(logl, numpy.ndarray):
            logl = logl.item()
        return logl

    def apply_additional_constraints(self):
        # FIXME: these should really be applied in the prior, but hard to
        # do right now because of the special functions involved, so we'll
        # just apply them in the likelihood call
        # Following copied from STU/CustomPrior_GLS.py
        spacetime = self.star.spacetime
        # limit polar radius to be outside the Schwarzschild photon sphere
        R_p = 1.0 + spacetime.epsilon * (-0.788 + 1.030 * spacetime.zeta)
        if R_p < 1.505 / spacetime.R_r_s:
            return False
        # 2-surface cross-section have a single maximum in |z|
        # i.e., an elliptical surface; minor effect on support, if any,
        # only for high spin frequencies
        mu = numpy.sqrt(
            -1.0 /
            (3*spacetime.epsilon*(-0.788 + 1.030 * spacetime.zeta)))
        if mu < 1.0:
            return False
        # check effective gravity at pole (where it is maximum) and
        # at equator (where it is minimum) are in NSX limits
        grav = xpsi.surface_radiation_field.effective_gravity(
            numpy.array([1.0, 0.0]),
            numpy.array([spacetime.R] * 2 ),
            numpy.array([spacetime.zeta] * 2),
            numpy.array([spacetime.epsilon] * 2))
        for g in grav:
            if not 13.7 <= g <= 15.0:
                return False
        return True

    @classmethod
    def from_config(cls, cp, **kwargs):
        # get the standard init args
        args = cls._init_args_from_config(cp)
        # get what instruments to analyze
        section = 'model'
        instruments = shlex.split(cp.get(section, 'instruments'))
        num_energies = int(cp.get(section, 'num-energies'))
        # get alpha values
        # FIXME: these should come from the prior
        section = 'model'
        min_alpha = float(cp.get(section, 'min-alpha'))
        max_alpha = float(cp.get(section, 'max-alpha'))
        alpha_bounds = (min_alpha, max_alpha)
        # load the instruments
        interstellar = interstellar_from_config(cp)
        signals = []
        nicer = None
        if 'nicer' in instruments:
            nicer = nicer_from_config(cp, interstellar, alpha_bounds)
            signals.append(nicer.signal)
        if 'xmm' in instruments:
            # XMM-PN
            xmm_pn = xmm_pn_from_config(cp, interstellar, alpha_bounds)
            signals.append(xmm_pn.signal)
            # for the other xmm instruments
            # FIXME: this is wonky. I think they did this so that the
            # alpha parameter is shared between instruments.
            from xpsi.Parameter import Derive
            class derive(Derive):
                global xmm_pn

                def __init__(self):
                    pass

                def __call__(self, boundto, caller=None):
                    return xmm_pn.instrument['alpha']
            derived_values = derive()
            # XMM-MOS1
            xmm_mos1 = xmm_mos1_from_config(cp, interstellar,
                                            derived_values=derived_values)
            signals.append(xmm_mos1.signal)
            # XMM-MOS2
            xmm_mos2 = xmm_mos2_from_config(cp, interstellar,
                                            derived_values=derived_values)
            signals.append(xmm_mos2.signal)
        # load the spacetime
        spacetime = spacetime_from_config(cp)
        # load the hotregions
        hotregions = hotregions_from_config(cp)
        photosphere = photosphere_from_config(cp, hotregions, spacetime.f)
        star = xpsi.Star(spacetime=spacetime,
                         photospheres=photosphere)
        args['star'] = star
        args['signals'] = [signals]
        args['num_energies'] = num_energies
        args.update(kwargs)
        return cls(**args)


#-----------------------------------------------------------------------------
#
#                   Helper functions/classes
#
#-----------------------------------------------------------------------------

class Instrument:
    """Generic class for storing instrument properties."""
    def __init__(self, data, instrument, signal):
        self.data = data
        self.instrument = instrument
        self.signal = signal


def interstellar_from_config(cp):
    section = 'interstellar'
    attenuation_path = cp.get(section, 'attenuation-path')
    column_density = (float(cp.get(section, 'min-column-density')),
                      float(cp.get(section, 'max-column-density')))
    interstellar = CustomInterstellar.from_SWG(
        attenuation_path,
        bounds=dict(column_density=column_density))
    return interstellar


def signal_from_config(cp, data, instrument, interstellar, **kwargs):
    section = 'signal'
    workspace_intervals = int(cp.get(section, 'workspace-intervals'))
    epsrel = float(cp.get(section, 'epsrel'))
    epsilon = float(cp.get(section, 'epsilon'))
    sigmas = float(cp.get(section, 'sigmas'))
    signal = CustomSignal(
        data=data,
        instrument=instrument,
        interstellar=interstellar,
        cache=False,
        workspace_intervals=workspace_intervals,
        epsrel=epsrel, epsilon=epsilon, sigmas=sigmas, **kwargs)
    return signal


def nicer_from_config(cp, interstellar, alpha_bounds):
    section = 'nicer'
    counts = numpy.loadtxt(cp.get(section, 'matrix-path'))
    # channels
    min_channel = int(cp.get(section, 'min-channels'))
    max_channel = int(cp.get(section, 'max-channels'))
    channels = numpy.arange(min_channel, max_channel)
    # phase bins
    phmin = float(cp.get(section, 'min-phases'))
    phmax = float(cp.get(section, 'max-phases'))
    phbins = int(cp.get(section, 'phases-nbins'))
    phases = numpy.linspace(phmin, phmax, phbins)
    # other stuff
    first = int(cp.get(section, 'first'))
    last = int(cp.get(section, 'last'))
    exposure_time = float(cp.get(section, 'exposure-time'))
    # load the data
    data = xpsi.Data(counts,
                     channels=channels,
                     phases=phases,
                     first=first,
                     last=last,
                     exposure_time=exposure_time)
    # load the instrument specs
    # files
    arf = cp.get(section, 'arf-path')
    rmf = cp.get(section, 'rmf-path')
    # channels
    min_input = int(cp.get(section, 'min-input'))
    max_input = int(cp.get(section, 'max-input'))
    channel_edges = cp.get(section, 'channels-path')
    # load the instrument
    instrument = CustomInstrument.NICER_XTI(bounds={'alpha': alpha_bounds},
                                            values={},
                                            ARF=arf,
                                            RMF=rmf,
                                            max_input=max_input,
                                            min_input=min_input,
                                            min_channel=min_channel,
                                            max_channel=max_channel,
                                            channel_edges=channel_edges,
                                            prefix='XTI')
    # load the signal
    signal = signal_from_config(cp, data, instrument, interstellar)
    return Instrument(data, instrument, signal)


def handle_XMM_event_list(event_list_path, instrument):
    """Event list with channel given in eV in the last column. """
    events = numpy.loadtxt(event_list_path, dtype=int,
                           skiprows=3, usecols=-1)
    spectrum = numpy.zeros(len(instrument.channels),
                           dtype=numpy.double)
    for event in events:
        for i in range(len(instrument.channel_edges) - 1):
            if instrument.channel_edges[i] <= event/1.0e3 < instrument.channel_edges[i+1]:
                spectrum[i] += 1.0
    return spectrum.reshape(-1,1) # count spectrum


def xmm_pn_from_config(cp, interstellar, alpha_bounds):
    section = 'xmm_pn'
    arf = cp.get(section, 'arf-path')
    rmf = cp.get(section, 'rmf-path')
    # channels
    min_channel = int(cp.get(section, 'min-channels'))
    max_channel = int(cp.get(section, 'max-channels'))
    channel_edges = cp.get(section, 'channels-path')
    min_input = int(cp.get(section, 'min-input'))
    max_input = int(cp.get(section, 'max-input'))
    # load the instrument
    instrument = CustomInstrument.XMM_PN(
        bounds={'alpha': alpha_bounds},
        values={},
        ARF=arf,
        RMF=rmf,
        min_input=min_input,
        max_input=max_input,
        min_channel=min_channel,
        max_channel=max_channel,
        channel_edges=channel_edges,
        prefix='PN')
    channels = instrument.channels
    # data setup
    spectrum_path = cp.get(section, 'spectrum-path')
    first = int(cp.get(section, 'first'))
    if 'last' in cp.options(section):
        last = int(cp.get(section, 'last'))
    else:
        last = len(channels) - 1
    phmin = float(cp.get(section, 'min-phases'))
    phmax = float(cp.get(section, 'max-phases'))
    phases = numpy.array([phmin, phmax])
    exposure_time = float(cp.get(section, 'exposure-time'))
    counts = handle_XMM_event_list(spectrum_path, instrument)
    data = xpsi.Data(counts,
                     channels=channels,
                     phases=phases,
                     first=first,
                     last=last,
                     exposure_time=exposure_time)
    # load the spectrum
    background_path = cp.get(section, 'background-path')
    # FIXME: not sure if skiprows and usecols should be
    # hardcoded or a config option
    spectrum = numpy.loadtxt(background_path,
                             skiprows=3, usecols=1,
                             dtype=numpy.double
                            )[min_channel:max_channel]

    # set support (copied directly from STU/main.py)
    support = numpy.zeros((len(spectrum), 2), dtype=numpy.double)
    support[:,0] = spectrum - 4.0 * numpy.sqrt(spectrum)
    support[support[:,0] < 0.0, 0] = 0.0
    support[:,1] = spectrum + 4.0 * numpy.sqrt(spectrum)
    for i in range(support.shape[0]):
        if support[i,1] == 0.0:
            for j in range(i, support.shape[0]):
                if support[j,1] > 0.0:
                    support[i,0] = support[j,1]
                    break
    support *= 0.9212 * (data.exposure_time / 4.51098e5) # BACKSCAL x exposure ratio
    support /= data.exposure_time # need count rate, so divide by exposure time
    # setup the signal model
    signal = signal_from_config(cp, data, instrument, interstellar,
                                support=support)
    return Instrument(data, instrument, signal)


def xmm_mos1_from_config(cp, interstellar, alpha_bounds=None,
                         derived_values=None):
    section = 'xmm_mos1'
    arf = cp.get(section, 'arf-path')
    rmf = cp.get(section, 'rmf-path')
    # channels
    min_channel = int(cp.get(section, 'min-channels'))
    max_channel = int(cp.get(section, 'max-channels'))
    channel_edges = cp.get(section, 'channels-path')
    min_input = int(cp.get(section, 'min-input'))
    max_input = int(cp.get(section, 'max-input'))
    bounds = {'alpha': alpha_bounds}
    if derived_values is not None:
        values = {'alpha': derived_values}
    instrument = CustomInstrument.XMM_MOS1(
        bounds=bounds,
        values=values,
        ARF=arf,
        RMF=rmf,
        min_input=min_input,
        max_input=max_input,
        min_channel=min_channel,
        max_channel=max_channel,
        channel_edges=channel_edges,
        prefix='MOS1')
    channels = instrument.channels
    # data setup
    spectrum_path = cp.get(section, 'spectrum-path')
    first = int(cp.get(section, 'first'))
    if 'last' in cp.options(section):
        last = int(cp.get(section, 'last'))
    else:
        last = len(channels) - 1
    phmin = float(cp.get(section, 'min-phases'))
    phmax = float(cp.get(section, 'max-phases'))
    phases = numpy.array([phmin, phmax])
    exposure_time = float(cp.get(section, 'exposure-time'))
    counts = handle_XMM_event_list(spectrum_path, instrument)
    data = xpsi.Data(counts,
                     channels=channels,
                     phases=phases,
                     first=first,
                     last=last,
                     exposure_time=exposure_time)
    # load the spectrum
    background_path = cp.get(section, 'background-path')
    # FIXME: not sure if skiprows and usecols should be
    # hardcoded or a config option
    spectrum = numpy.loadtxt(background_path,
                             skiprows=3, usecols=1,
                             dtype=numpy.double
                            )[min_channel:max_channel]
    # set support (copied directly from STU/main.py)
    support = numpy.zeros((len(spectrum), 2), dtype=numpy.double)
    support[:,0] = spectrum - 4.0 * numpy.sqrt(spectrum)
    support[support[:,0] < 0.0, 0] = 0.0
    support[:,1] = spectrum + 4.0 * numpy.sqrt(spectrum)
    for i in range(support.shape[0]):
        if support[i,1] == 0.0:
            for j in range(i, support.shape[0]):
                if support[j,1] > 0.0:
                    support[i,0] = support[j,1]
                    break
    support *= 1.074 * (data.exposure_time / 1.57623e6) # BACKSCAL x exposure ratio
    support /= data.exposure_time # need count rate, so divide by exposure time
    # setup the signal model
    signal = signal_from_config(cp, data, instrument, interstellar,
                                support=support)
    return Instrument(data, instrument, signal)


def xmm_mos2_from_config(cp, interstellar, alpha_bounds=None,
                         derived_values=None):
    section = 'xmm_mos2'
    arf = cp.get(section, 'arf-path')
    rmf = cp.get(section, 'rmf-path')
    # channels
    min_channel = int(cp.get(section, 'min-channels'))
    max_channel = int(cp.get(section, 'max-channels'))
    channel_edges = cp.get(section, 'channels-path')
    min_input = int(cp.get(section, 'min-input'))
    max_input = int(cp.get(section, 'max-input'))
    # get alpha bounds
    bounds = {'alpha': alpha_bounds}
    if derived_values is not None:
        values = {'alpha': derived_values}
    # initialize the instrument
    instrument = CustomInstrument.XMM_MOS2(
        bounds=bounds,
        values=values,
        ARF=arf,
        RMF=rmf,
        min_input=min_input,
        max_input=max_input,
        min_channel=min_channel,
        max_channel=max_channel,
        channel_edges=channel_edges,
        prefix='MOS2')
    channels = instrument.channels
    # data setup
    spectrum_path = cp.get(section, 'spectrum-path')
    first = int(cp.get(section, 'first'))
    if 'last' in cp.options(section):
        last = int(cp.get(section, 'last'))
    else:
        last = len(channels) - 1
    phmin = float(cp.get(section, 'min-phases'))
    phmax = float(cp.get(section, 'max-phases'))
    phases = numpy.array([phmin, phmax])
    exposure_time = float(cp.get(section, 'exposure-time'))
    counts = handle_XMM_event_list(spectrum_path, instrument)
    data = xpsi.Data(counts,
                     channels=channels,
                     phases=phases,
                     first=first,
                     last=last,
                     exposure_time=exposure_time)
    # load the spectrum
    background_path = cp.get(section, 'background-path')
    # FIXME: not sure if skiprows and usecols should be
    # hardcoded or a config option
    spectrum = numpy.loadtxt(background_path,
                             skiprows=3, usecols=1,
                             dtype=numpy.double
                            )[min_channel:max_channel]
    # set support (copied directly from STU/main.py)
    support = numpy.zeros((len(spectrum), 2), dtype=numpy.double)
    support[:,0] = spectrum - 4.0 * numpy.sqrt(spectrum)
    support[support[:,0] < 0.0, 0] = 0.0
    support[:,1] = spectrum + 4.0 * numpy.sqrt(spectrum)
    for i in range(support.shape[0]):
        if support[i,1] == 0.0:
            for j in range(i, support.shape[0]):
                if support[j,1] > 0.0:
                    support[i,0] = support[j,1]
                    break
    support *= 1.260 * (data.exposure_time / 1.51256e6) # BACKSCAL x exposure ratio
    support /= data.exposure_time # need count rate, so divide by exposure time
    # setup the signal model
    signal = signal_from_config(cp, data, instrument, interstellar,
                                support=support)
    return Instrument(data, instrument, signal)


def spacetime_from_config(cp):
    # setup the spacetime
    section = 'spacetime'
    # FIXME: these should be pulled from the prior
    spacetime_bounds = {
        'mass': (None, None),
        'radius': (3.0*gravradius(1.0), 16.0),
        'distance': (None, None),
        'cos_inclination': (None, None),
    }
    spacetime_freq = float(cp.get(section, 'frequency'))
    return xpsi.Spacetime(spacetime_bounds,
                          {'frequency': spacetime_freq})


def read_hotspot_args(cp, section, common=None):
    if common is None:
        out = {}
    else:
        out = common.copy()
    # boolean options
    boolopts = ['symmetry', 'omit', 'cede', 'concentric',
                'is_antiphased']
    for opt in boolopts:
        out[opt] = cp.has_option(section, opt.replace('_', '-'))
    # options to cast to integer
    intopts = ['sqrt_num_cells',
               'min_sqrt_num_cells',
               'max_sqrt_num_cells',
               'num_leaves',
               'num_rays',
               'image_order_limit']
    # all other options will be read as string
    for opt in cp.options(section):
        storeopt = opt.replace('-', '_')
        if storeopt in boolopts:
            continue
        val = cp.get(section, opt)
        if storeopt in intopts:
            val = int(val)
        out[storeopt] = val
    return out


def hotregions_from_config(cp):
    # setup the hotspots
    # get the number and common options
    section = 'hotspots'
    spottags = cp.get_subsections(section)
    # FIXME: these should come from the prior
    hotspots_bounds = {
        'p': dict(
            super_colatitude = (0.001, numpy.pi-0.001),
            super_radius = (0.001, numpy.pi/2.0 - 0.001),
            phase_shift = (-0.5, 0.5),
            super_temperature = (5.1, 6.8)),
        's': dict(
            super_colatitude = (0.001, numpy.pi - 0.001),
            super_radius = (0.001, numpy.pi/2.0 - 0.001),
            phase_shift = (-0.5, 0.5),
            super_temperature = (5.1, 6.8)
        )
    }
    # get the common options
    common_opts = read_hotspot_args(cp, section)
    # load the spots
    spots = []
    for tag in spottags:
        spotopts = read_hotspot_args(cp, '-'.join([section, tag]),
                                     common=common_opts)
        bnds = hotspots_bounds[tag]
        spots.append(xpsi.HotRegion(bounds=bnds,
                                    values={},
                                    prefix=tag,
                                    **spotopts))
    return xpsi.HotRegions(tuple(spots))


def photosphere_from_config(cp, hotregions, spacetime_freq):
    # load the photosphere
    section = 'photosphere'
    photosphere = CustomPhotosphere(
        hot=hotregions, elsewhere=None,
        values={'mode_frequency': spacetime_freq})
    photosphere.hot_atmosphere = cp.get(section, 'atmosphere-path')
    return photosphere
