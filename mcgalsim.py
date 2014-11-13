from copy import deepcopy
from argparse import ArgumentParser
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import galsim
import emcee
import triangle
from astropy.utils.console import ProgressBar

def walker_ball(alpha, spread, nwalkers):
    return [alpha+np.random.randn(len(alpha))*spread for i in xrange(nwalkers)]

def lnprior(p):
    x0, y0, n, flux, HLR, e1, e2 = p
    if n < 0.3 or n > 6.2: return -np.inf
    if flux < 0: return -np.inf
    if HLR < 0.1: return -np.inf
    if e1**2 + e2**2 > 1.0: return -np.inf
    return 0.0

def lnprob(p, target_img, noise_var, psftype):
    lp = lnprior(p)
    if not np.isfinite(lp):
        return -np.inf
    x0, y0, n, flux, HLR, e1, e2 = p
    model_img = galsim.ImageD(25, 25, scale=0.2)

    gal = galsim.Sersic(n=n, half_light_radius=HLR, flux=flux)
    gal = gal.shear(e1=e1, e2=e2)
    gal = gal.shift(x0, y0)

    if psftype == 'atmopt':
        lam_over_diam = 700e-9 / 8.4 * 3600 * 180.0 / np.pi
        aberrations = [0.0]*4 + [0.1]*8
        obscuration = 0.5
        atm_FWHM = 0.6
        atm_e1 = 0.01
        atm_e2 = 0.02
        psf = galsim.Convolve(galsim.Kolmogorov(fwhm=atm_FWHM).shear(e1=atm_e1, e2=atm_e2),
                              galsim.OpticalPSF(lam_over_diam=lam_over_diam,
                                                aberrations=aberrations,
                                                obscuration=obscuration))
    elif psftype == 'moffat':
        psf = galsim.Moffat(fwhm = 0.7, beta=3.0).shear(e1=0.01, e2=0.02)

    final = galsim.Convolve(gal, psf)
    try:
        final.drawImage(image=model_img)
    except:
        return -np.inf
    lnlike = -0.5 * np.sum((target_img.array - model_img.array)**2/noise_var)

    return lp + lnlike

def autocorrtime(chain):
    N, K = chain.shape
    if N < 100: return
    M = 50
    tau_int = emcee.autocorr.integrated_time(chain, window=M)
    while (M < 4*tau_int.max()) & (M < N/4) :
        M = 4.1*tau_int.max()
        tau_int = emcee.autocorr.integrated_time(chain, window=M)
    return tau_int

def mcgalsim(args):
    bd = galsim.BaseDeviate(args.seed)
    gn = galsim.GaussianNoise(bd)

    target_img = galsim.ImageD(25, 25, scale=0.2)

    gal = galsim.Sersic(n=args.n,
                        half_light_radius=args.HLR,
                        flux=args.flux)
    gal = gal.shear(e1=args.e1, e2=args.e2)
    gal = gal.shift(args.x0, args.y0)

    if args.psf == 'atmopt':
        lam_over_diam = 700e-9 / 8.4 * 3600 * 180.0 / np.pi
        aberrations = [0.0]*4 + [0.1]*8
        obscuration = 0.5
        atm_FWHM = 0.6
        atm_e1 = 0.01
        atm_e2 = 0.02
        psf = galsim.Convolve(galsim.Kolmogorov(fwhm=atm_FWHM).shear(e1=atm_e1, e2=atm_e2),
                              galsim.OpticalPSF(lam_over_diam=lam_over_diam,
                                                aberrations=aberrations,
                                                obscuration=obscuration))
    elif args.psf == 'moffat':
        psf = galsim.Moffat(fwhm = 0.7, beta=3.0).shear(e1=0.01, e2=0.02)

    final = galsim.Convolve(gal, psf)
    final.drawImage(image=target_img)
    noise_var = target_img.addNoiseSNR(gn, args.snr, preserve_flux=True)

    p1 = [args.x0, args.y0, args.n, args.flux, args.HLR, args.e1, args.e2]
    dp1 = 0.001
    ndim = len(p1)
    p0 = walker_ball(p1, dp1, args.nwalkers)
    # print np.array(p0).mean(axis=0)

    sampler = emcee.EnsembleSampler(args.nwalkers, ndim, lnprob,
                                    args=(target_img, noise_var, args.psf),
                                    threads=args.nthreads)
    pp, lnp, rstate = sampler.run_mcmc(p0, 1)
    sampler.reset()

    lnps = []
    dts = []

    print "sampling"
    with ProgressBar(args.nburn+args.nsamples) as bar:
        for i in range(args.nburn+args.nsamples):
            bar.update()
            t1 = time.time()
            pp, lnp, rstate = sampler.run_mcmc(pp, 1, lnprob0=lnp, rstate0=rstate)
            dt = (time.time() - t1) / args.nwalkers * args.nthreads
            lnps.append(deepcopy(lnp))
            dts.append(dt)
    samples = sampler.chain
    lnps = np.array(lnps)
    dts = np.array(dts)
    flat_samples = samples[:, args.nburn:, :].reshape((-1, ndim)) # flat_samples excludes burn-in
    if flat_samples.shape[0] > 2000:
        tau_int = autocorrtime(flat_samples)
    print "making triangle plot"
    fig = triangle.corner(flat_samples, labels=["x0", "y0", "n", "flux", "HLR", "e1", "e2"],
                          truths=[args.x0, args.y0, args.n, args.flux, args.HLR, args.e1, args.e2])
    fig.savefig("triangle.png", dpi=220)

    print "making walker plot"
    # Try to make plot aspect ratio near golden
    nparam = ndim+2 # add 2 for lnp and dt
    ncols = int(np.ceil(np.sqrt(nparam*1.6)))
    nrows = int(np.ceil(1.0*nparam/ncols))

    fig = plt.figure(figsize = (3.0*ncols,3.0*nrows))
    for i, p in enumerate(["x0", "y0", "n", "flux", "HLR", "e1", "e2"]):
        ax = fig.add_subplot(nrows, ncols, i+1)
        ax.plot(samples[..., i].T)
        ax.set_ylabel(p)
        ax.set_yticklabels(ax.get_yticks(), rotation=45)
        ax.set_xticklabels(ax.get_xticks(), rotation=45)
        ylim = ax.get_ylim()
        ylim = np.r_[ylim[0], (ylim[1]-ylim[0])*1.1+ylim[0]]
        ax.set_ylim(ylim)
        xlim = ax.get_xlim()
        ax.set_xlim(xlim)
        ax.fill_between([args.nburn,xlim[1]], [ylim[0]]*2, [ylim[1]]*2, color="#CCCCCC")
        if "tau_int" in locals():
            ax.text(0.6, 0.90, r"$\tau_\mathrm{{int}} = {:g}$".format(int(tau_int[i])),
                    transform=ax.transAxes)

    # lnp chains
    ax = fig.add_subplot(nrows, ncols, i+2)
    ax.plot(lnps)
    ax.set_ylabel(r"ln(prob)")
    ax.set_yticklabels(ax.get_yticks(), rotation=45)
    ax.set_xticklabels(ax.get_xticks(), rotation=45)
    xlim = ax.get_xlim()
    ax.set_xlim(xlim)
    ylim = ax.get_ylim()
    ax.set_ylim(ylim)
    ax.fill_between([args.nburn,xlim[1]], [ylim[0]]*2, [ylim[1]]*2, color="#CCCCCC")

    # delta t distribution
    ax = fig.add_subplot(nrows, ncols, i+3)
    ax.hist(dts)
    ax.set_xlabel(r"$\Delta t (s)$")
    ax.set_ylabel(r"#")
    ax.set_yticklabels(ax.get_yticks(), rotation=45)
    ax.set_xticklabels(ax.get_xticks(), rotation=45)

    fig.tight_layout()
    fig.savefig("walkers.png", dpi=220)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--x0', type=float, default=0.0,
                        help="Galaxy centroid (default: 0.0)")
    parser.add_argument('--y0', type=float, default=0.0,
                        help="Galaxy centroid (default: 0.0)")
    parser.add_argument('-n', type=float, default=1.0,
                        help="Galaxy Sersic index (default: 1.0)")
    parser.add_argument('--flux', type=float, default=10.0,
                        help="Galaxy flux (default: 10.0)")
    parser.add_argument('--HLR', type=float, default=0.5,
                        help="Galaxy half light radius (default: 0.5)")
    parser.add_argument('--e1', type=float, default=0.0,
                        help="Galaxy ellipticity (default: 0.0)")
    parser.add_argument('--e2', type=float, default=0.0,
                        help="Galaxy ellipticity (default: 0.0)")
    parser.add_argument('--snr', type=float, default=80.0,
                        help="Signal-to-noise ratio (default: 80.0)")
    parser.add_argument('--psf', default='moffat',
                        help="psf type: (moffat | atmopt) (default: moffat)")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random number seed (default: 0)")
    parser.add_argument('--nwalkers', type=int, default=32,
                        help="Number of walkers (default: 32)")
    parser.add_argument('--nburn', type=int, default=30,
                        help="Numbers of burn-in samples (default: 30)")
    parser.add_argument('--nsamples', type=int, default=30,
                        help="Numbers of samples per walker (default: 30)")
    parser.add_argument('--nthreads', type=int, default=4,
                        help="Numbers of threads (default: 4)")
    args = parser.parse_args()
    mcgalsim(args)
