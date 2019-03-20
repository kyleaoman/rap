import numpy as np
import matplotlib.pyplot as pp
import emcee
import corner
from itertools import product
import os


class Rap(object):

    def __init__(self, model=None):
        self.model = model
        self.ndim = model.ndim
        self.results = None
        return

    def _exec_fit(self, pool=None):
        kwargs = {'pool': pool} if pool is not None else {}
        self._sampler = emcee.EnsembleSampler(
            self.nwalkers,
            self.ndim,
            self.model.logProbability,
            **kwargs
        )
        self._sampler.run_mcmc(self.guess, self.niter, progress=True)

    def fit(self, guess, niter=5500, burn=500, parallel=True):
        self.guess = guess
        self.nwalkers = len(guess)
        self.niter = niter
        self.burn = burn
        if parallel:
            from multiprocessing import Pool
            # prevent interference with parallel
            omp_num_threads = os.environ['OMP_NUM_THREADS']
            os.environ['OMP_NUM_THREADS'] = '1'
            with Pool() as pool:
                self._exec_fit(pool=pool)
            os.environ['OMP_NUM_THREADS'] = omp_num_threads
        else:
            self._exec_fit()
        self.results = dict()
        getter_kwargs = dict(flat=True, discard=burn)
        self.results['samples'] = self._sampler.get_chain(**getter_kwargs)
        self.results['lnL'] = self._sampler.get_log_prob(**getter_kwargs)
        self.results['theta_ml'] = self.results['samples'][
            np.argmax(self.results['lnL'])
        ]
        return

    def cornerfig(self, save=None, save_format='pdf', fignum=None,
                  labels=None, truths=None):
        if self.results is None:
            raise RuntimeError('Run fit before trying to create plot.')
        if labels is None:
            labels = [''] * self.ndim
        cornerfig = pp.figure(fignum)
        pp.clf()
        axes = {}
        for spn in range(self.ndim ** 2):
            axes[(spn // self.ndim, spn % self.ndim)] = \
                cornerfig.add_subplot(self.ndim, self.ndim, spn + 1)
        corner.corner(
            self.results['samples'],
            labels=labels,
            fig=cornerfig,
            truths=truths,
            levels=1.0 - np.exp(-0.5 * np.array([1, 2, 3]) ** 2)
        )
        for row, col in product(range(self.ndim), range(self.ndim)):
            pp.sca(axes[(row, col)])
            if row == col:  # diagonal
                pp.axvline(
                    x=self.results['theta_ml'][col],
                    color='red',
                    ls='solid'
                )
            elif col < row:  # lower left
                x = self.results['theta_ml'][col],
                y = self.results['theta_ml'][row],
                pp.plot(x, y, marker='*', mec='red', mfc='red', ls='None')
        if save is not None:
            pp.savefig(save, format=save_format)
        return
