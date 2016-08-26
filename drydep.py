# -*- coding: utf-8 -*-

"""
Calculation of the dry deposition velocity of aerosol particles on the surface
of a vegetation.

In CFD models, the deposition of the aerosol may be included in the concentration
equation by an additional term,
dc/dt = - LAD ud c,
where c is the mass concentration, LAD is one-sided leaf area density,
and ud is the deposition velocity as calculated here.
This package serves to calculate ud, given the state of the atmosphere, the
properties of the vegetation, and the properties of the particles.
"""

from math import exp, pi, sqrt, log
import numpy as np

# Constants
KB = 1.3806488e-23   # Boltzmann constant
LAMBD = 0.066*1e-6   # Mean free path in air
GRAV = 9.81          # Gravitational constant

def slip_correction_factor(dp):
    """Slip correction factor for gravitational settling of particle with
    diameter 'dp'"""
    return 1. + 2*LAMBD/dp*(1.257 + 0.4*exp(-1.1*dp/(2*LAMBD)))

def browndif_coeff(environ, particle):
    """Coefficient of Brownian diffusion"""
    T = environ["T"]
    mua = environ["mu"]
    dp = particle["d"]
    Cc = slip_correction_factor(dp)
    DB = KB*T*Cc/(3*pi*mua*dp)
    return DB

def settling_velocity(environ, particle):
    """Return settling velocity of the particle in the air."""
    rhop = particle["rho"]
    dp = particle["d"]
    mua = environ["mu"]
    Cc = slip_correction_factor(dp)
    return (rhop*(dp**2)*GRAV*Cc)/(18*mua)

class DepVelModel():
    """Base class for deposition velocity models"""

    @classmethod
    def get_components(cls):
        """Return list of names of deposition velocity components"""
        return []

    def get(self, comp):
        """Return value of the specified component"""
        return getattr(self, comp)()


class PetroffModel(DepVelModel):
    """Vegetation model following (Petroff et al., 2008) for conifer trees and
    (Petroff et al., 2009) for broadleaves.

    In both cases, Dirac leaf size distribution is assumed.

    Parameters:
        environ: 'mu', 'rho', 'u', 'uf'
        particle: 'd', 'rho'
        vegetation: 'de', 'leafType', 'leafDist'
    """
    def __init__(self, environ, particle, vegetation):
        self.mua = environ["mu"]
        self.rhoa = environ["rho"]
        self.nua = self.mua/self.rhoa
        self.u = environ["u"]
        self.uf = environ["uf"]

        self._set_veget_params(vegetation)

        self.dp = particle["d"]
        self.rhop = particle["rho"]

        # Schmidt number
        self.Sc = self.nua/browndif_coeff(environ, particle)

        # Settling velocity
        self.us = settling_velocity(environ, particle)

    @classmethod
    def get_components(cls):
        return ["total", "brownian_diffusion", "sedimentation", "interception",
                "impaction", "turbulent_impaction"]

    def _set_veget_params(self, vegetation):
        self.de = vegetation["de"]
        self.leaf_type = vegetation["leafType"]
        leaf_dist = vegetation["leafDist"]

        dist_needle = {"horizontal":  (2./pi**2, 1./pi),
                       "plagiophile": (0.27,     0.22),
                       "vertical":    (1./pi,    0.),
                       "uniform":     (0.27,     2./pi**2)}

        dist_broadleaf = {"horizontal":  (0.,      0.5),
                          "plagiophile": (0.216,   0.340),
                          "vertical":    (1./pi,   0.),
                          "uniform":     (2/pi**2, 1./pi)}

        if self.leaf_type == "needle":
            self.kx, self.kz = dist_needle[leaf_dist]
            self.Cb = 0.467
            self.beta = 0.6
        elif self.leaf_type == "broadleaf":
            self.kx, self.kz = dist_broadleaf[leaf_dist]
            self.Cb = 0.664
            self.beta = 0.47
        else:
            raise ValueError("Unknown leafType %s" % self.leaf_type)

    def brownian_diffusion(self):
        """Brownian diffusion"""
        Re = self.u*self.de/self.nua
        # See the article for constants
        nb = 0.5
        return 2 * self.u*self.Cb*(self.Sc**(-2./3.))*(Re**(nb - 1))

    def sedimentation(self):
        """Sedimentation"""
        return 2 * self.kz*self.us

    def interception(self):
        """Interception"""
        if self.leaf_type == "needle":
            return 2 * (2*self.u*self.kx*(self.dp/self.de))
        elif self.leaf_type == "broadleaf":
            return 2 * (self.u*self.kx/2.0*(self.dp/self.de)
                        *(2.0 + log(4.0*self.de/self.dp)))
        else:
            raise ValueError("Unknown leafType %s" % self.leaf_type)

    def impaction(self):
        """Impaction"""
        Cc = slip_correction_factor(self.dp)
        tauP = self.rhop*Cc*(self.dp**2)/(18.0*self.mua)
        St = self.u*tauP/self.de
        IIm = (St/(St + self.beta))**2

        vIm = self.u*self.kx*IIm
        return 2 * vIm

    def turbulent_impaction(self):
        """Turbulent impaction"""
        kIT1 = 3.5e-4
        kIT2 = 0.18

        Cc = slip_correction_factor(self.dp)
        tauP = self.rhop*Cc*(self.dp**2)/(18.0*self.mua)

        tauPPlus = tauP * self.uf**2 / self.nua

        if tauPPlus <= 20:
            return 2 * self.uf * kIT1 * tauPPlus**2
        else:
            return 2 * self.uf * kIT2

    def total(self):
        """Total deposition velocity"""
        return (self.brownian_diffusion()
                + self.sedimentation()
                + self.interception()
                + self.impaction()
                + self.turbulent_impaction())

class ApprxPetroffModel(DepVelModel):
    """Approximated Petroff model described in (Sip, 2016).

    Parameters:
        environ: 'mu', 'rho', 'u', 'uf'
        particle: 'd', 'rho'
        vegetation: 'de', 'leafType', 'leafDist'
    """

    def __init__(self, environ, particle, vegetation):
        self.mua = environ["mu"]
        self.rhoa = environ["rho"]
        self.nua = self.mua/self.rhoa
        self.u = environ["u"]
        self.uf = environ["uf"]

        self.set_veget_params(vegetation)

        self.dp = particle["d"]
        self.rhop = particle["rho"]

        # Settling velocity
        self.us = settling_velocity(environ, particle)

        self.environ = environ
        self.particle = particle

        sti = np.array([1e-2, 1e-1, 1e0, 1e1, 1e2])
        ei = (sti/(sti + self.beta))**2
        self.iim_fun = lambda x: np.interp(x, sti, ei, left=0.0, right=1.0)

    @classmethod
    def get_components(cls):
        return ["total", "brownian_diffusion", "sedimentation", "interception",
                "impaction", "turbulent_impaction"]

    def set_veget_params(self, vegetation):
        self.de = vegetation["de"]
        self.leaf_type = vegetation["leafType"]
        leaf_dist = vegetation["leafDist"]

        dist_needle = {"horizontal":  (2./pi**2, 1./pi),
                       "plagiophile": (0.27,     0.22),
                       "vertical":    (1./pi,    0.),
                       "uniform":     (0.27,     2./pi**2)}

        dist_broadleaf = {"horizontal":  (0.,      0.5),
                          "plagiophile": (0.216,   0.340),
                          "vertical":    (1./pi,   0.),
                          "uniform":     (2/pi**2, 1./pi)}

        if self.leaf_type == "needle":
            self.kx, self.kz = dist_needle[leaf_dist]
            self.Cb = 0.467
            self.beta = 0.6
        elif self.leaf_type == "broadleaf":
            self.kx, self.kz = dist_broadleaf[leaf_dist]
            self.Cb = 0.664
            self.beta = 0.47
        else:
            raise ValueError("Unknown leafType %s" % self.leaf_type)

    def brownian_diffusion(self):
        """Brownian diffusion"""
        Re = self.u*self.de/self.nua
        nb = 0.5
        Cc = 3.34*LAMBD/self.dp
        DB = (KB*self.environ["T"]*Cc)/(3*pi*self.mua*self.dp)
        Sc = self.nua/DB
        return 2*self.u*self.Cb*(Sc**(-2./3.))*(Re**(nb - 1))

    def sedimentation(self):
        """Sedimentation"""
        return 2*self.kz*self.us

    def interception(self):
        """Interception"""
        if self.leaf_type == "needle":
            return 2*(2*self.u*self.kx*(self.dp/self.de))
        elif self.leaf_type == "broadleaf":
            a = 4.57
            b = -0.078
            return 2*(self.u*self.kx/2.0*(self.dp/self.de)
                      *(2.0 + log(4.0*self.de) + a*self.dp**b))
        else:
            raise ValueError("Unknown leafType %s" % self.leaf_type)

    def impaction(self):
        """Impaction"""
        Cc = 1.0
        tauP = self.rhop*Cc*(self.dp**2)/(18.0*self.mua)
        St = self.u*tauP/self.de
        IIm = self.iim_fun(St)
        vIm = self.u*self.kx*IIm
        return 2*vIm

    def turbulent_impaction(self):
        """Turbulent impaction"""
        kIT1 = 3.5e-4
        kIT2 = 0.18

        Cc = slip_correction_factor(self.dp)
        tauP = self.rhop*Cc*(self.dp**2)/(18.0*self.mua)

        tauPPlus = tauP * self.uf**2 / self.nua

        if tauPPlus <= 20:
            return 2 * self.uf * kIT1 * tauPPlus**2
        else:
            return 2 * self.uf * kIT2

    def total(self):
        """Total deposition velocity"""
        return (self.brownian_diffusion()
                + self.sedimentation()
                + self.interception()
                + self.impaction()
                + self.turbulent_impaction())


class RaupachModel(DepVelModel):
    """Deposition velocity model used in (Raupach et al, 2001).

    Incorporates only impaction and Brownian diffusion.
    Parameters:
        environ: 'mu', 'rho', 'u'
        particle: 'd', 'rho'
        vegetation: 'de', 'frontalToTotal'
    """

    def __init__(self, environ, particle, vegetation):
        self.mua = environ["mu"]
        self.rhoa = environ["rho"]
        self.nua = self.mua/self.rhoa
        self.u = environ["u"]

        self.de = vegetation["de"]
        self.frontal_to_total = vegetation["frontalToTotal"]

        self.dp = particle["d"]
        self.rhop = particle["rho"]

        # Schmidt number
        self.Sc = self.nua/browndif_coeff(environ, particle)

    @classmethod
    def get_components(cls):
        return ['brownian_diffusion', 'impaction', 'total']

    def brownian_diffusion(self):
        """Brownian diffusion"""
        Cpol = 1.32  # Polhausen coeffient
        Re = self.u*self.de/self.nua
        gpb = 2*self.u*Cpol*(Re**-0.5)*(self.Sc**(-2./3.))
        return gpb

    def impaction(self):
        """Impaction"""
        St = ((self.dp**2)*self.rhop*self.u)/(18*self.mua*self.de)
        p = 0.8
        q = 2.0
        E = (St/(St + p))**q
        return 2*self.u*self.frontal_to_total*E

    def total(self):
        """Total deposition velocity"""
        return self.brownian_diffusion() + self.impaction()

class BruseModel(DepVelModel):
    """Vegetation model as described in (Bruse, 2007).

    Incorporates sedimentation, impaction and Brownian diffusion."""

    def __init__(self, environ, particle, vegetation):
        self.mua  = environ["mu"]
        self.rhoa = environ["rho"]
        self.nua = self.mua/self.rhoa
        self.u = environ["u"]

        self.set_veget_params(vegetation)

        self.dp = particle["d"]
        self.rhop = particle["rho"]

        # Schmidt number
        self.Sc = self.nua/browndif_coeff(environ, particle)

        # Settling velocity
        self.us = settling_velocity(environ, particle)

        self.ra = self.A*sqrt(self.de/max(self.u, 0.05))
        self.ustar = sqrt(self.u/self.ra)

    def set_veget_params(self, vegetation):
        self.de = vegetation["de"]
        if vegetation["leafType"] == "broadleaf":
            self.A = 87.0
        else:
            self.A = 200.0

    @classmethod
    def get_components(cls):
        return ["total", "brownian_diffusion", "sedimentation", "impaction"]

    def sedimentation(self):
        """Sedimentation"""
        return self.us

    def impaction(self):
        """Impaction"""
        St = (self.us*(self.ustar**2))/(GRAV*self.nua)
        return self.ustar*(10.0**(-3.0/St))

    def brownian_diffusion(self):
        """Brownian diffusion"""
        return self.ustar*(self.Sc**(-2.0/3.0))

    def total(self):
        """Total deposition velocity"""
        rb = 1.0/(self.brownian_diffusion() + self.impaction())
        return 1.0/(self.ra + rb + self.ra*rb*self.us) + self.us


# Available models
MODELS = {
    "Petroff": PetroffModel,
    "ApprxPetroff": ApprxPetroffModel,
    "Raupach": RaupachModel,
    "Bruse": BruseModel
}

def calc(model, environ, particle, vegetation, component="total"):
    """Calculate deposition using given model.

    All models are initialized by three dicts, describing the atmospheric
    environment, particle, and vegetation. Used nomenclature is given below.
    All quantities are in SI units.
    Not all parameter are used in all models, check the documentation of each
    model for the necessary ones.

    Args:
        environ (dict): Description of the atmospheric environment.
            Dict may include:
                'mu': Dynamic viscosity.
                'rho': Air density.
                'T': Air temperature.
                'u': Wind speed.
                'uf': Friction velocity.
        particle (dict): Description of the particle.
            Dict may include:
                'd': Diameter of the particle.
                'rho': Density of the particle.
        vegetation (dict): Description of the vegetation.
            Dict may include:
                'de': Vegetation element (leaf, needle) diameter.
                'leafType': Either 'needle' or 'broadleaf'
                'leafDist': Distribution of the leaf orientation. One of
                    'horizontal', 'plagiophile', 'vertical', 'uniform'.
                'frontalToTotal': Fraction of frontal area to total
                    (two sided) leaf area.
        component (str): Component of the deposition velocity associated with a
            physical process to get, or 'total' for the total deposition velocity.
            For each model, :py:meth:`DepVelModel.get_components()` returns
            list of the components ('total' included).
    """

    if model not in MODELS:
        raise ValueError("Model " + model + " not implemented.")

    return MODELS[model](environ, particle, vegetation).get(component)
