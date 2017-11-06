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

    def __init__(self, observations, interp='lanczos15', nointerp_psf=False, target_psf=None):
        """
        nointerp_psf=True means no interpolation
        """
        self.observations = observations
        self.target_psf=target_psf
        self.interp = interp
        self.nointerp_psf=nointerp_psf

        # use a nominal sky position
        self.sky_center = galsim.CelestialCoord(5*galsim.hours, -25*galsim.degrees)

        self._add_obs()

    def get_mean_coadd(self):
        """
        perform a weight mean coaddition
        """

        self._set_coadded_image()
        self._set_coadded_psf()
        self._set_coadded_weight_map()
        self._set_coadd_jacobian_cen()

        self.coadd_obs.update_meta_data(self.observations.meta)

        return self.coadd_obs
    
    def _set_coadded_weight_map(self):
        """
        coadd the noise image realizations and take var

        also set the .noise attribute
        """
        coadd_obs=self.coadd_obs
        weights=self.weights

        weight_map = np.zeros(coadd_obs.weight.shape)

        ny,nx = coadd_obs.image.shape

        wcs = coadd_obs.jacobian.get_galsim_wcs()

        coadd_noise = galsim.Sum([image*w for image,w in zip(self.noise_images,weights)])
        coadd_noise_image = galsim.Image(nx, ny, wcs=wcs)
        coadd_noise.drawImage(image=coadd_noise_image, method='no_pixel')

        weight_map[:,:] = 1./np.var(coadd_noise_image.array)
        coadd_obs.set_weight(weight_map)

        coadd_obs.noise = coadd_noise_image.array

    def _set_coadd_jacobian_cen(self):
        """
        set the center

        currently only support the canonical center
        """
        cen = self.canonical_center
        self.coadd_obs.jacobian.set_cen(
            row=cen.y-1,
            col=cen.x-1,
        )


    def _set_coadded_image(self):
        """
        do the actual coadding, with appropriate weights
        """
        coadd_obs=self.coadd_obs
        weights=self.weights

        wcs = coadd_obs.jacobian.get_galsim_wcs()

        coadd = galsim.Sum([image*w for image,w in zip(self.images,weights)])

        ny,nx = coadd_obs.image.shape

        coadd_image = galsim.Image(nx, ny, wcs=wcs)
        coadd.drawImage(image=coadd_image, method='no_pixel')

        coadd_obs.set_image(coadd_image.array)

    def _set_coadded_psf(self):
        """
        set the coadd psf
        """
        coadd_obs=self.coadd_obs
        weights=self.weights

        if self.nointerp_psf:
            coadd_psf_image = self.psfs[0].copy()
            coadd_psf_image[:,:] = 0.0
            for psf,w in zip(self.psfs, weights):
                coadd_psf_image += w*psf

        else:
            psf_ny,psf_nx = coadd_obs.psf.image.shape
            psf_wcs = coadd_obs.psf.jacobian.get_galsim_wcs()

            coadd_psf = galsim.Sum([psf*w for psf,w in zip(self.psfs,weights)])
            coadd_psf_image = galsim.Image(psf_nx, psf_ny, wcs=psf_wcs)
            coadd_psf.drawImage(image=coadd_psf_image, method='no_pixel')

            coadd_psf_image = coadd_psf_image.array

        coadd_obs.psf.set_image(coadd_psf_image)

    def _set_coadd_obs(self):
        """
        base the coadd off the observation with largest
        postage stamp
        """
        nxs=np.zeros(len(self.observations))
        nys=nxs.copy()

        for i,obs in enumerate(self.observations):
            if self.select_obs(obs) is False:
                continue
            ny,nx = obs.image.shape
            nxs[i] = nx
            nys[i] = ny

        #argx = nxs.argmin()
        #argy = nys.argmin()
        argx = nxs.argmax()
        argy = nys.argmax()

        assert argx==argy

        nx = nxs[argx]
        ny = nys[argy]

        self.coadd_obs = copy.deepcopy(self.observations[argx])

        tim = galsim.ImageD(nx,ny)
        self.canonical_center = tim.trueCenter()


    def _add_obs(self):
        """
        add observations as interpolated images

        also keep track of psfs, variances, and noise realizations
        """
        self.images = []
        self.psfs = []
        self.vars = np.zeros(len(self.observations))
        self.noise_images = []

        self._set_coadd_obs()

        for i,obs in enumerate(self.observations):

            if self.select_obs(obs) is False:
                continue

            offset_pixels = obs.meta['offset_pixels']
            if offset_pixels is None:
                xoffset, yoffset = 0.0, 0.0
            else:
                xoffset, yoffset = offset_pixels

            offset = galsim.PositionD(xoffset, yoffset)
            image_center = self.canonical_center + offset

            # interplated image, shifted to center of the postage stamp
            wcs = galsim.TanWCS(
                affine=galsim.AffineTransform(
                    obs.jacobian.dudcol,
                    obs.jacobian.dudrow,
                    obs.jacobian.dvdcol,
                    obs.jacobian.dvdrow,
                    origin=image_center,
                ),
                world_origin=self.sky_center,
            )
            psf_wcs = galsim.TanWCS(
                affine=galsim.AffineTransform(
                    obs.jacobian.dudcol,
                    obs.jacobian.dudrow,
                    obs.jacobian.dvdcol,
                    obs.jacobian.dvdrow,
                    origin=self.canonical_center,
                ),
                world_origin=self.sky_center,
            )

            image = galsim.InterpolatedImage(
                galsim.Image(obs.image,wcs=wcs),
                offset=offset,
                x_interpolant=self.interp,
            )

            psf_image = obs.psf.image.copy()
            psf_image /= psf_image.sum()

            if self.nointerp_psf:
                psf = psf_image
            else:
                psf = galsim.InterpolatedImage(
                    galsim.Image(psf_image,wcs=psf_wcs),
                    x_interpolant=self.interp,
                )

            if self.target_psf is not None:
                raise NotImplementedError("need to normalize psf "
                                          "correctly so weight maps "
                                          "are consistent")

                psf_inv = galsim.Deconvolve(psf)
                matched_image = galsim.Convolve([image, psf_inv, self.target_psf])
                self.images.append(matched_image)
            else:
                self.images.append(image)

            # normalize psf
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

        self.weights = 1./self.vars
        self.weights /= self.weights.sum()

    def select_obs(self, observation):
        return True


