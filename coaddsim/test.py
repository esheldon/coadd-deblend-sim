from .coadd import CoaddImages

try:
    import nsim
except ImportError:
    pass

EXAMPLE_CONF = {
    'images': {
        'noise': 0.35,
        'object': {
            'cen_shift': {
                'type': 'uniform',
                'radius': 20,
            },
            'nepoch': 100,
            'stamp_size': [48, 48],
        },
        'psf': {
            's2n': 10000,
            'stamp_size': [48, 48],
        },
        'wcs': {
            'dudx': 0.0,
            'dudy': 0.263,
            'dvdx': 0.263,
            'dvdy': 0.0,
        }
    },
    'object': {
        'model': 'bdk',
        'fracdev': {
            'type': 'uniform',
            'range': [0.0, 1.0],
        },
        'g': {
            'type': 'ba',
            'sigma': 0.2,
        },
        'r50_flux': {
            'type': 'cosmos',
        },
    },
    'psf': {
        'beta': 3.5,
        'model': 'moffat',
        'r50': 0.54,
        'shape': [0.0, 0.0],
    },
    'shear': [0.02, 0.0],
}

def test(seed=2342, nepoch=100):
    coadder = CoaddImages()

    conf={}
    conf.update(EXAMPLE_CONF)
    conf['images']['object']['nepoch'] = nepoch

    sim = nsim.sime.Sim(conf, seed) 

    obslist = sim()


    coadder.add_obs(obslist)
    coadd_obs = coadder.get_mean_coadd()

    return coadd_obs
