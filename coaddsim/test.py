from __future__ import print_function
try:
    raw_input
except:
    raw_input=input
    xrange=range

from .coadd import CoaddImages

try:
    import nsim
except ImportError:
    pass


# this is an nsim configuration file
_confstring="""
# min s/n approximately 10
images:

    noise: 0.10

    psf:
        s2n: 10000
        stamp_size: [48,48] # optional

    object:
        nepoch: 10

        stamp_size: [48,48]

        cen_shift:
            type: "uniform"
            radius: 1 # pixels
            #radius: 0 # pixels

    wcs:
      dudx: 0.0
      dudy: 0.263
      dvdx: 0.263
      dvdy: 0.0

psf:
    model: "moffat"

    beta: 3.5
    # same as fwhm = 0.9
    r50: 0.54
        #type: "uniform"
        #range: [0.41,0.6]

    shape: [0.0, 0.0]


object:
    model: "bdk"

    g:
        type: "ba"
        sigma: 0.2

    r50_flux:
        type: "cosmos"


    # fraction of flux in the bulge
    fracdev:
        type: "uniform"
        range: [0.0, 1.0]

shear: [0.02, 0.00]
"""

def getconf():
    import yaml
    return yaml.load(_confstring)

def test(seed=2342,
         num=1,
         nepoch=10,
         doplot=False,
         run_admom=False):

    conf=getconf()
    conf['images']['object']['nepoch'] = nepoch

    sim = nsim.sime.Sim(conf, seed) 

    for i in xrange(num):
        obslist = sim()

        coadder = CoaddImages(obslist)
        coadd_obs = coadder.get_mean_coadd()
        coadd_obs = coadder.get_mean_coadd()


        if run_admom:
            import ngmix
            jac = coadd_obs.jacobian
            Tguess=4.0*jac.get_scale()**2
            #Tguess=0.5

            for i in xrange(4):
                fitter=ngmix.admom.run_admom(coadd_obs, Tguess)
                res=fitter.get_result()
                if res['flags'] == 0:
                    print("    admom s2n:",res['s2n'])
                    print("    admom pars:",res['pars'])

                    if doplot:
                        import images
                        mim=fitter.get_gmix().make_image(
                            coadd_obs.image.shape,
                            jacobian=jac,
                        )
                        images.multiview(mim, title='model')
                    break
                else:
                    print("    admom failed",res['flags'],res['flagstr'])

        if doplot:
            import images
            images.multiview(coadd_obs.image, title='coadd')
            if num > 1:
                if raw_input('hit a key (q to quit): ')=='q':
                    return

