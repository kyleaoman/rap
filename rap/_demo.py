import numpy as np
import rap.models._demomodel as model
from rap import Rap
import matplotlib.pyplot as pp
from matplotlib.backends.backend_pdf import PdfPages


def demo():
    atrue = 1
    btrue = 0
    truths = (atrue, btrue)
    model.xi = np.linspace(0, 1, 10)
    model.yi = (atrue * model.xi + btrue) + (np.random.rand(10) - .5) * .1

    R = Rap(model)
    guess = [np.array([1, 0]) + np.random.rand(2) * .01 for i in range(10)]
    R.fit(guess, niter=5500, burn=500, parallel=True)
    with PdfPages('demo.pdf') as pdffile:
        R.cornerfig(save=pdffile, save_format='pdf', fignum=1,
                    labels=['a', 'b'], truths=truths)
        pp.figure(2)
        pp.plot(model.xi, model.yi, 'ok')
        abest, bbest = R.results['theta_ml']
        ys = abest * model.xi + bbest
        pp.plot(model.xi, ys, ls='solid', color='slategray')
        yt = atrue * model.xi + btrue
        pp.plot(model.xi, yt, ls='solid', color='red')
        pp.xlabel('x')
        pp.ylabel('y')
        pp.savefig(pdffile, format='pdf')
