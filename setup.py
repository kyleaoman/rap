from setuptools import setup

setup(
    name='rap',
    version='0.1',
    description='Wrapper for emcee MCMC analysis.',
    url='',
    author='Kyle Oman',
    author_email='koman@astro.rug.nl',
    license='',
    packages=['rap'],
    install_requires=['numpy', 'emcee'],
    include_package_data=True,
    zip_safe=False
)
