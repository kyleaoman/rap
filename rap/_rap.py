import os
import numpy as np
import emcee
from itertools import product


class Rap(object):

    def __init__(self, model=None):
        self.model = model
        self.ndim = model.ndim
        self.results = None
        return

    def _exec_fit(self, pool=None):
        kwargs = {}
        if pool is not None:
            kwargs['pool'] = pool
        try:
            kwargs['blobs_dtype'] = self.model.blobs_dtype
        except AttributeError:
            pass
        self._sampler = emcee.EnsembleSampler(
            self.nwalkers,
            self.ndim,
            self.model.logProbability,
            **kwargs
        )
        self._sampler.run_mcmc(self.guess, self.niter, progress=True)

    def fit(self, guess, niter=5500, burn=500, parallel=8):
        self.guess = guess
        self.nwalkers = len(guess)
        self.niter = niter
        self.burn = burn
        if parallel > 1:
            from multiprocessing import Pool
            # prevent interference with parallel from e.g. numpy
            omp_num_threads = os.environ['OMP_NUM_THREADS']
            os.environ['OMP_NUM_THREADS'] = '1'
            with Pool(processes=parallel) as pool:
                self._exec_fit(pool=pool)
            os.environ['OMP_NUM_THREADS'] = omp_num_threads
        elif parallel == 1:
            self._exec_fit()
        else:
            raise ValueError('Use parallel=int (number of cores).')
        self.results = dict()
        getter_kwargs = dict(flat=True, discard=burn)
        self.results['thetas'] = self._sampler.get_chain(**getter_kwargs)
        self.results['blobs'] = self._sampler.get_blobs(**getter_kwargs)
        self.results['lnL'] = self._sampler.get_log_prob(**getter_kwargs)
        self.results['theta_ml'] = self.results['thetas'][
            np.argmax(self.results['lnL'])
        ]
        self.results['theta_perc_16_50_84'] = list(zip(
            *np.percentile(self.results['thetas'], [16, 50, 84], axis=0)))
        return

    def cornerfig(self, save=None, save_format='pdf', fignum=None,
                  labels=None, truths=None):
        import corner
        import matplotlib.pyplot as pp
        if self.results is None:
            raise RuntimeError('Run fit before trying to create plot.')
        if labels is None:
            labels = [''] * self.ndim
        cornerfig = pp.figure(fignum,
                              figsize=(1.5 * self.ndim, 1.5 * self.ndim))
        pp.clf()
        axes = dict()
        for spn in range(self.ndim ** 2):
            axes[(spn // self.ndim, spn % self.ndim)] = \
                cornerfig.add_subplot(self.ndim, self.ndim, spn + 1)
        corner.corner(
            self.results['thetas'],
            labels=labels,
            fig=cornerfig,
            truths=truths,
            levels=1.0 - np.exp(-0.5 * np.array([1, 2, 3]) ** 2),
            fill_contours=True,
            plot_datapoints=False,
            color='red',
            plot_density=False,
            hist_kwargs=dict(histtype='stepfilled', color='#FFCCCC',
                             ec='red'),
            quantiles=(.3173 / 2, .5, 1 - .3173 / 2),
            bins=25,
            smooth=(1., 1.)
        )
        for row, col in product(range(self.ndim), range(self.ndim)):
            pp.sca(axes[(row, col)])
            if row == col:  # diagonal
                pp.axvline(
                    x=self.results['theta_ml'][col],
                    color='#AA0000',
                    ls='solid'
                )
                try:
                    bounds = self.model.prior_bounds
                except AttributeError:
                    pass
                else:
                    pp.xlim(bounds[row])
            elif col < row:  # lower left
                x = self.results['theta_ml'][col],
                y = self.results['theta_ml'][row],
                pp.plot(x, y, marker='*', mec='#AA0000', mfc='#AA0000',
                        ls='None')
                try:
                    bounds = self.model.prior_bounds
                except AttributeError:
                    pass
                else:
                    pp.xlim(bounds[col])
                    pp.ylim(bounds[row])
            if col <= row:
                pp.tick_params(labelsize=8)
        if save is not None:
            pp.savefig(save, format=save_format)
        return axes

    def overplot_cornerfig(self, cornerfig, axes, save=None, save_format='pdf',
                           truths=None):
        import corner
        import matplotlib.pyplot as pp
        if self.results is None:
            raise RuntimeError('Run fit before trying to create plot.')
        corner.corner(
            self.results['thetas'],
            fig=cornerfig,
            truths=truths,
            levels=1.0 - np.exp(-0.5 * np.array([1, 2, 3]) ** 2),
            fill_contours=True,
            contourf_kwargs=dict(alpha=0),
            color='#92A2B2',
            plot_datapoints=False,
            plot_density=False,
            hist_kwargs=dict(histtype='step', color='#92A2B2'),
            quantiles=(.3173 / 2, .5, 1 - .3173 / 2),
            bins=25,
            smooth=(1., 1.)
        )
        for row, col in product(range(self.ndim), range(self.ndim)):
            pp.sca(axes[(row, col)])
            if row == col:  # diagonal
                pp.axvline(
                    x=self.results['theta_ml'][col],
                    color='slategray',
                    ls='solid'
                )
                try:
                    bounds = self.model.prior_bounds
                except AttributeError:
                    pass
                else:
                    pp.xlim(bounds[row])
            elif col < row:  # lower left
                x = self.results['theta_ml'][col],
                y = self.results['theta_ml'][row],
                pp.plot(x, y, marker='*', mec='slategray', mfc='slategray',
                        ls='None')
                try:
                    bounds = self.model.prior_bounds
                except AttributeError:
                    pass
                else:
                    pp.xlim(bounds[col])
                    pp.ylim(bounds[row])
        if save is not None:
            pp.savefig(save, format=save_format)
        return
