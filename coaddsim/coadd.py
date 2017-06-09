"""
convert a set of observations produced by
nsim (https://github.com/esheldon/nsim) to
a coadd using Galsim to do the interpolation
"""

import nsim
import ngmix
import numpy as np
import galsim
import copy

class CoaddImages():

    def __init__(self, observations, interp='lanczos3', target_psf=None):
        self.observations = observations
        self.target_psf=target_psf
        self.interp = interp

        self._add_obs()

    def get_mean_coadd(self, find_cen=False):
        """
        perform a weight mean coaddition

        parameters
        ----------
        find_cen: bool, optional
            If True, set the jacobian center based on a fit
        """

        weights = 1./self.vars
        weights /= weights.sum()

        coadd_obs = self.coadd_obs
        wcs = coadd_obs.jacobian.get_galsim_wcs()
        psf_wcs = coadd_obs.psf.jacobian.get_galsim_wcs()

        coadd = galsim.Sum([image*w for image,w in zip(self.images,weights)])

        ny,nx = coadd_obs.image.shape
        psf_ny,psf_nx = coadd_obs.psf.image.shape

        coadd_image = galsim.Image(nx, ny, wcs=wcs)
        coadd.drawImage(image=coadd_image)

        coadd_psf = galsim.Sum([psf*w for psf,w in zip(self.psfs,weights)])
        coadd_psf_image = galsim.Image(psf_nx, psf_ny, wcs=psf_wcs)
        coadd_psf.drawImage(image=coadd_psf_image)

        coadd_obs.set_image(coadd_image.array)
        coadd_obs.psf.set_image(coadd_psf_image.array)

        weight_map = np.zeros(self.coadd_obs.weight.shape)
        coadd_var = (self.vars*weights*weights).sum()
        weight_map[:,:] = 1./coadd_var
        self.coadd_obs.set_weight(weight_map)


        crow, ccol = self.canonical_center
        if find_cen:
            moments = galsim.hsm.FindAdaptiveMom(galsim.Image(self.coadd_obs.image))
            centroid = moments.moments_centroid

            shift = galsim.PositionD(ccol, crow) - centroid
            self.coadd_obs.jacobian.set_cen(col=shift.x, row=shift.y)
        else:
            # center the jacobian on the canonical center
            self.coadd_obs.jacobian.set_cen(row=crow, col=ccol)

        self.coadd_obs.update_meta_data(self.observations.meta)
        return self.coadd_obs


    def _add_obs(self):

        self.images = []
        self.psfs = []
        self.vars = np.zeros(len(self.observations))

        self.coadd_obs = None

        for i,obs in enumerate(self.observations):

            if self.select_obs(obs) is False:
                continue

            # currently using image size and scale of the first image for the coadd scale

            if self.coadd_obs is None:
                self.coadd_obs = copy.deepcopy(obs)
                self.canonical_center = (np.array(self.coadd_obs.image.shape)-1.0)/2.0

            # interplated image, shifted to center of the postage stamp
            wcs = obs.jacobian.get_galsim_wcs()

            image = galsim.InterpolatedImage(
                galsim.Image(obs.image,wcs=wcs),
                x_interpolant=self.interp,
            )
            psf = galsim.InterpolatedImage(
                galsim.Image(obs.psf.image,wcs=wcs),
                x_interpolant=self.interp,
            )

            yoffset, xoffset = obs.meta['offset_pixels']
            sky_shift = -wcs.toWorld(galsim.PositionD(x=xoffset, y=yoffset))

            image = image.shift(sky_shift)

            if self.target_psf is not None:
                raise NotImplementedError("need to normalize psf correctly so weight maps are consistent")

                psf_inv = galsim.Deconvolve(psf)
                matched_image = galsim.Convolve([image, psf_inv, self.target_psf])
                self.images.append(matched_image)
            else:
                self.images.append(image)

            # normalize psf
            obs.psf.image /= np.sum(obs.psf.image)
            self.psfs.append(psf)

            # assume variance is constant
            var = 1./obs.weight.max()

            self.vars[i] = var


    def select_obs(self, observation):
        return True


