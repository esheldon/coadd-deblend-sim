"""
convert a set of observations produced by
nsim (https://github.com/esheldon/nsim) to
a coadd using Galsim to do the interpolation
"""

import nsim
import ngmix
import numpy as np
import galsim

class CoaddImages():

    def __init__(self, interp='lanczos3', size=48, center=24.5):
        self.interp = interp
        self.images = []
        self.psfs = []
        self.vars = []
        self.fitters = []
        self.coadd_obs = None
        self.size = size
        self.center = center


    def add_obs(self, observations, target_psf=None):

        for obs in observations:

            if self.select_obs(obs) is False:
                continue

            # currently using scale of the first image for the coadd scale

            if self.coadd_obs is None:
                self.coadd_obs = obs
                self.scale = obs.jacobian.get_scale()

            # interplated image, shifted to center of the postage stamp
            #wcs = galsim.JacobianWCS(obs.jacobian.dudcol, obs.jacobian.dudrow, obs.jacobian.dvdcol, obs.jacobian.dvdrow)
            wcs = obs.jacobian.get_galsim_wcs()

            image = galsim.InterpolatedImage(galsim.Image(obs.image,wcs=wcs), x_interpolant=self.interp)
            psf = galsim.InterpolatedImage(galsim.Image(obs.psf.image,wcs=wcs), x_interpolant=self.interp)
            sky_shift = -wcs.toWorld(galsim.PositionD(*obs.meta['offset_pixels']))

            image = image.shift(sky_shift)

            if target_psf is not None:
                psf_inv = galsim.Deconvolve(psf)
                matched_image = galsim.Convolve([image, psf_inv, target_psf])
                self.images.append(matched_image)
            else:
                self.images.append(image)

            # normalize psf
            obs.psf.image /= np.sum(obs.psf.image)
            self.psfs.append(psf)

            # assume variance is constant
            var = 1./(np.mean(obs.weight))
            self.vars.append(var)

            # fit parameters
            fitter = ngmix.admom.run_admom(obs, 4*self.scale**2)
            self.fitters.append(fitter)
        self.vars = np.array(self.vars)

    def select_obs(self, observation):
        return True

    def get_mean_coadd(self):

        weights = 1./self.vars
        weights /= weights.sum()

        coadd = galsim.Sum([image*w for image,w in zip(self.images,weights)])
        coadd_image = galsim.Image(self.size, self.size, scale=self.scale)
        coadd.drawImage(image=coadd_image)

        coadd_psf = galsim.Sum([psf*w for psf,w in zip(self.psfs,weights)])
        coadd_psf_image = galsim.Image(self.size, self.size, scale=self.scale)
        coadd_psf.drawImage(image=coadd_psf_image)

        self.coadd_obs.image = coadd_image.array
        self.coadd_obs.psf.image = coadd_psf_image.array

        weight_map = np.zeros(self.coadd_obs.weight.shape)
        coadd_var = (self.vars*weights*weights).sum()
        weight_map[:,:] = 1./coadd_var
        self.coadd_obs.set_weight(weight_map)

        moments = galsim.hsm.FindAdaptiveMom(galsim.Image(self.coadd_obs.image))
        centroid = moments.moments_centroid
        shift = galsim.PositionD(self.center, self.center) - centroid

        self.coadd_obs.jacobian.set_cen(col=shift.x, row=shift.y)
        fitter = ngmix.admom.run_admom(self.coadd_obs, 4*self.scale**2)

        return self.coadd_obs

