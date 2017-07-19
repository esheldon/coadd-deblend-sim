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

    def __init__(self, observations, interp='lanczos15', target_psf=None):
        self.observations = observations
        self.target_psf=target_psf
        self.interp = interp
        # use a nominal sky position
        self.sky_center = galsim.CelestialCoord(5*galsim.hours, -25*galsim.degrees)

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
        coadd.drawImage(image=coadd_image, method='no_pixel')

        coadd_psf = galsim.Sum([psf*w for psf,w in zip(self.psfs,weights)])
        coadd_psf_image = galsim.Image(psf_nx, psf_ny, wcs=psf_wcs)
        coadd_psf.drawImage(image=coadd_psf_image, method='no_pixel')

        coadd_obs.set_image(coadd_image.array)
        coadd_obs.psf.set_image(coadd_psf_image.array)

        weight_map = np.zeros(self.coadd_obs.weight.shape)

        coadd_noise = galsim.Sum([image*w for image,w in zip(self.noise_images,weights)])
        coadd_noise_image = galsim.Image(nx, ny, wcs=wcs)
        coadd_noise.drawImage(image=coadd_noise_image, method='no_pixel')

        weight_map[:,:] = 1./np.var(coadd_noise_image.array)
        self.coadd_obs.set_weight(weight_map)

        cen = self.canonical_center
        if find_cen:
            try:
                moments = galsim.hsm.FindAdaptiveMom(galsim.Image(self.coadd_obs.image))
                cen = moments.moments_centroid
            except:
                pass

        self.coadd_obs.jacobian.set_cen(
            row=cen.y,
            col=cen.x,
        )

        self.coadd_obs.update_meta_data(self.observations.meta)
        self.coadd_obs.noise = coadd_noise_image.array
        return self.coadd_obs


    def _add_obs(self):

        self.images = []
        self.psfs = []
        self.vars = np.zeros(len(self.observations))
        self.noise_images = []

        self.coadd_obs = None

        for i,obs in enumerate(self.observations):

            if self.select_obs(obs) is False:
                continue

            # currently using image size and scale of the first image for the coadd scale

            if self.coadd_obs is None:
                self.coadd_obs = copy.deepcopy(obs)
                ny,nx = obs.image.shape
                tim = galsim.ImageD(nx,ny)
                self.canonical_center = tim.trueCenter()

            offset_pixels = obs.meta['offset_pixels']
            if offset_pixels is None:
                xoffset, yoffset = 0.0, 0.0
            else:
                xoffset, yoffset = offset_pixels

            offset = galsim.PositionD(xoffset, yoffset)
            image_center = self.canonical_center + offset

            # interplated image, shifted to center of the postage stamp
            wcs = galsim.TanWCS(affine=galsim.AffineTransform(obs.jacobian.dudcol, obs.jacobian.dudrow,
                                                              obs.jacobian.dvdcol, obs.jacobian.dvdrow,
                                                              origin=image_center),
                                world_origin=self.sky_center)
            psf_wcs = galsim.TanWCS(affine=galsim.AffineTransform(obs.jacobian.dudcol, obs.jacobian.dudrow,
                                                                  obs.jacobian.dvdcol, obs.jacobian.dvdrow,
                                                                  origin=self.canonical_center),
                                    world_origin=self.sky_center)

            image = galsim.InterpolatedImage(
                galsim.Image(obs.image,wcs=wcs),
                offset=offset,
                x_interpolant=self.interp,
            )
            psf = galsim.InterpolatedImage(
                galsim.Image(obs.psf.image,wcs=psf_wcs),
                x_interpolant=self.interp,
            )

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

            # create noise image given the variance
            noise = galsim.Image(*obs.image.shape,wcs=wcs)
            noise.addNoise(galsim.GaussianNoise(sigma=np.sqrt(var)))

            noise_image = galsim.InterpolatedImage(
                noise,
                offset=offset,
                x_interpolant=self.interp,
            )
            self.noise_images.append(noise_image)


    def select_obs(self, observation):
        return True


