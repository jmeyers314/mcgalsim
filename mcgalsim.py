from argparse import ArgumentParser

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import galsim
import emcee
import triangle
from astropy.utils.console import ProgressBar

def walker_ball(alpha, spread, nwalkers):
    return [alpha+(np.random.rand(len(alpha))*spread-0.5*spread) for i in xrange(nwalkers)]

def lnprior(p):
    x0, y0, n, flux, HLR, e1, e2 = p
    if n < 0.3 or n > 6.2: return -np.inf
    if flux < 0: return -np.inf
    if HLR < 0.1: return -np.inf
    if e1**2 + e2**2 > 1.0: return -np.inf
    return 0.0

def lnprob(p, psf_args, target_img, noise_var):
    lp = lnprior(p)
    if not np.isfinite(lp):
        return -np.inf
    x0, y0, n, flux, HLR, e1, e2 = p
    beta, PSF_FWHM, PSF_e1, PSF_e2 = psf_args
    model_img = galsim.ImageD(25, 25, scale=0.2)

    gal = galsim.Sersic(n=n, half_light_radius=HLR, flux=flux)
    gal = gal.shear(e1=e1, e2=e2)
    gal = gal.shift(x0, y0)

    psf = galsim.Moffat(beta=beta,
                        fwhm=PSF_FWHM)
    psf = psf.shear(e1=PSF_e1, e2=PSF_e2)

    final = galsim.Convolve(gal, psf)
    final.drawImage(image=model_img)
    lnlike = -0.5 * np.sum((target_img.array - model_img.array)**2/noise_var)

    return lp + lnlike

def mcgalsim(args):
    bd = galsim.BaseDeviate(args.seed)
    gn = galsim.GaussianNoise(bd)

    target_img = galsim.ImageD(25, 25, scale=0.2)

    gal = galsim.Sersic(n=args.n,
                        half_light_radius=args.HLR,
                        flux=args.flux)
    gal = gal.shear(e1=args.e1, e2=args.e2)
    gal = gal.shift(args.x0, args.y0)

    psf = galsim.Moffat(beta=args.beta,
                        fwhm=args.PSF_FWHM)
    psf = psf.shear(e1=args.PSF_e1, e2=args.PSF_e2)

    final = galsim.Convolve(gal, psf)
    final.drawImage(image=target_img)
    noise_var = target_img.addNoiseSNR(gn, args.snr, preserve_flux=True)

    psf_args = [args.beta, args.PSF_FWHM, args.PSF_e1, args.PSF_e2]
    p1 = [args.x0, args.y0, args.n, args.flux, args.HLR, args.e1, args.e2]
    dp1 = np.array([0.01, 0.01, 0.01, 10.0, 0.1, 0.01, 0.01])
    ndim = len(p1)
    p0 = walker_ball(p1, dp1, args.nwalkers)

    sampler = emcee.EnsembleSampler(args.nwalkers, ndim, lnprob,
                                    args=(psf_args, target_img, noise_var),
                                    threads=args.nthreads)
    pp, lnp, rstate = sampler.run_mcmc(p0, 5)
    sampler.reset()

    with ProgressBar(args.nburn+args.nsamples) as bar:
        for i in range(args.nburn+args.nsamples):
            bar.update()
            pp, lnp, rstate = sampler.run_mcmc(pp, 1, lnprob0=lnp, rstate0=rstate)
    samples = sampler.chain[:, args.nburn:, :].reshape((-1, ndim))
    print "making triangle plot"
    fig = triangle.corner(samples, labels=["x0", "y0", "n", "flux", "HLR", "e1", "e2"],
                          truths=[args.x0, args.y0, args.n, args.flux, args.HLR, args.e1, args.e2])
    fig.savefig("mcgalsim.png", dpi=300)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--x0', type=float, default=0.0,
                        help="Galaxy centroid")
    parser.add_argument('--y0', type=float, default=0.0,
                        help="Galaxy centroid")
    parser.add_argument('-n', type=float, default=1.0,
                        help="Galaxy Sersic index")
    parser.add_argument('--flux', type=float, default=1000.0,
                        help="Galaxy flux")
    parser.add_argument('--HLR', type=float, default=0.5,
                        help="Galaxy half light radius")
    parser.add_argument('--e1', type=float, default=0.0,
                        help="Galaxy ellipticity")
    parser.add_argument('--e2', type=float, default=0.0,
                        help="Galaxy ellipticity")
    parser.add_argument('--beta', type=float, default=3.0,
                        help="PSF Moffat index")
    parser.add_argument('--PSF_FWHM', type=float, default=0.6,
                        help="PSF FWHM")
    parser.add_argument('--PSF_e1', type=float, default=0.0,
                        help="PSF ellipticity")
    parser.add_argument('--PSF_e2', type=float, default=0.0,
                        help="PSF ellipticity")
    parser.add_argument('--snr', type=float, default=80.0,
                        help="Signal-to-noise ratio")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random number seed")
    parser.add_argument('--nwalkers', type=int, default=32,
                        help="Number of walkers")
    parser.add_argument('--nburn', type=int, default=30,
                        help="Numbers of burn-in samples")
    parser.add_argument('--nsamples', type=int, default=30,
                        help="Numbers of samples per walker")
    parser.add_argument('--nthreads', type=int, default=8,
                        help="Numbers of threads")
    args = parser.parse_args()
    mcgalsim(args)
