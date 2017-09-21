"""
An example how to simulate a 3D n_pred / n_obs cube.

Using CTA IRFs.

TODOs:
* For `compute_npred_cube` we're getting ``NaN`` values where flux == 0.
  This shouldn't happen and is bad. Figure out what's going on and fix!

"""
#from __future__ import absolute_import, division, print_function, unicode_literals
import astropy.units as u
import numpy as np
import yaml
import pyfits
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, Angle
from gammapy.image import SkyImage
from configuration import make_ref_cube
from astropy.io import fits
from sherpa.astro.ui import *
from gammapy.cube import SkyCube
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion
from astropy.table import Table

from gammapy.spectrum.models import LogParabola
from gammapy.spectrum.models import PowerLaw
from gammapy.spectrum.models import ExponentialCutoffPowerLaw
from gammapy.spectrum import SpectrumFit
from gammapy.spectrum import SpectrumObservation
from gammapy.spectrum import PHACountsSpectrum
from gammapy.irf import  EffectiveAreaTable, EnergyDispersion, EffectiveAreaTable2D, EnergyDispersion2D
from gammapy.irf import EnergyDependentMultiGaussPSF

from gammapy.background import FOVCube

def read_config(filename):
    with open(filename) as fh:
        config = yaml.load(fh)
    
    # TODO: fix the following issue in a better way (e.g. raise an error or fix somehow)
    # apparently this gets returned as string, but we want float!?
    # prefactor1: 2e-12
    config['model']['prefactor'] = float(config['model']['prefactor'])

    return config


def get_irfs(config):
    filename = 'irf_file.fits'
    
    offset = Angle(config['selection']['offset_fov'] * u.deg)

    
    psf_fov = EnergyDependentMultiGaussPSF.read(filename, hdu='POINT SPREAD FUNCTION')
    psf = psf_fov.to_energy_dependent_table_psf(theta=offset)
    
    print(' psf', psf)
    aeff = EffectiveAreaTable2D.read(filename, hdu='EFFECTIVE AREA')
    
    edisp_fov = EnergyDispersion2D.read(filename, hdu='ENERGY DISPERSION')
    edisp = edisp_fov.to_energy_dispersion(offset=offset)
    table = fits.open('irf_file.fits')['BACKGROUND']
    table.columns.change_name(str('BGD'), str('Bgd'))
    table.header['TUNIT7'] = '1 / (MeV s sr)'
    bkg = FOVCube.from_fits_table(table, scheme='bg_cube')
    
    # TODO: read background once it's working!
    # bkg = Background3D.read(filename, hdu='BACKGROUND')
    
    return dict(psf=psf, aeff=aeff, edisp=edisp, bkg=bkg)

def region_mask(self, region):
    """Create a boolean cube mask for a region.
        
    The mask is:
        
    - ``True`` for pixels inside the region
    - ``False`` for pixels outside the region
        
    Parameters
    ----------
    region : `~regions.PixelRegion` or `~regions.SkyRegion`
    Region in pixel or sky coordinates.
    
    Returns
    -------
    mask : `SkyCube`
    A boolean sky cube mask.
    """
    mask = self.sky_image_ref.region_mask(region)
    data = mask.data * np.ones(self.data.shape, dtype='bool') * u.Unit('')
    wcs = self.wcs.deepcopy() if self.wcs else None
    return self.__class__(name=self.name, data=data.astype('bool'), wcs=wcs,
                          energy_axis=self.energy_axis)



def cube_sed(cube, mask=None, flux_type='differential', counts=None,
             errors=False, standard_error=0.1, spectral_index=2.3):
    """Creates SED from SkyCube within given lat and lon range.
        
    Parameters
    ----------
    cube : `~gammapy.data.SkyCube`
    Spectral cube of either differential or integral fluxes (specified
    with flux_type)
    mask : array_like, optional
    2D mask array, matching spatial dimensions of input cube.
    A mask value of True indicates a value that should be ignored,
    while a mask value of False indicates a valid value.
    flux_type : {'differential', 'integral'}
    Specify whether input cube includes differential or integral fluxes.
    counts :  `~gammapy.data.SkyCube`, optional
    Counts cube to allow Poisson errors to be calculated. If not provided,
    a standard_error should be provided, or zero errors will be returned.
    errors : bool
    If True, computes errors, if possible, according to provided inputs.
    If False (default), returns all errors as zero.
    standard_error : float
    If counts cube not provided, but error values required, this specifies
    a standard fractional error to be applied to values. Default = 0.1.
    spectral_index : float
    If integral flux is provided, this is used to calculate differential
    fluxes and energies (according to the Lafferty & Wyatt model-based
    method, assuming a power-law model).
    
    Returns
    -------
    table : `~astropy.table.Table`
    A spectral energy table of energies, differential fluxes and
    differential flux errors. Units as those input.
    """
    
    #lon, lat = cube.spatial_coordinate_images
    
    values = []
    for i in np.arange(cube.data.shape[0]):
        if mask is None:
            bin = cube.data[i].sum()
        else:
            #print cube.data[i][mask.data[i]].sum()
            bin = cube.data[i][mask.data[i]].sum()
        values.append(bin.value)
    values = np.array(values)

    if errors:
        if counts is None:
            # Counts cube required to calculate poisson errors
            errors = np.ones_like([values]) * standard_error
        else:
            errors = []
            for i in np.arange(counts.data.shape[0]):
                if mask is None:
                    bin = counts.data[i].sum()
                else:
                    bin = counts.data[i][mask].sum()
                r_error = 1. / (np.sqrt(bin.value))
                errors.append(r_error)
            errors = np.array([errors])
    else:
        errors = np.zeros_like([values])
    
    if flux_type == 'differential':


        energy = cube.energies('center')
        table = Table()
        table['ENERGY'] = energy,
        table['DIFF_FLUX'] = u.Quantity(values, cube.data.unit),
        table['DIFF_FLUX_ERR'] = u.Quantity(errors * values, cube.data.unit)

    elif flux_type == 'integral':
    
        emins = cube.energies('edges')[:-1]
        emaxs = cube.energies('edges')[1:]
        table = compute_differential_flux_points(x_method='lafferty',
                                                 y_method='power_law',
                                                 spectral_index=spectral_index,
                                                 energy_min=emins, energy_max=emaxs,
                                                 int_flux=values,
                                                 int_flux_err=errors * values)
    
    else:
        raise ValueError('Unknown flux_type: {0}'.format(flux_type))

    return table






def main():
    
    #################################
    
    #filename = 'flux_cube.fits'
    #cube = SkyCube.read(filename)
    #pos = SkyCoord(83.6333 * u.deg, 22.0144 * u.deg, frame='icrs')
    #on_size = 1.742 * u.deg #0.167
    #on_region = CircleSkyRegion(pos, on_size)
    #mask = cube.region_mask(on_region)

    #sed_table = cube_sed(cube, mask=mask)
    
    #import csv
    
    #csv_file = open('sed_table.csv', 'wb')
    #csv_writer = csv.writer(csv_file,
    #                        delimiter=' ',
    #                        quotechar='"',
    #                        quoting=csv.QUOTE_MINIMAL)

    #csv_writer.writerow(['E(TeV)','DIFF_FLUX (1/(cm2 s sr TeV))','DIFF_FLUX_ERR'])
    #for i in range(len(sed_table[0][0])):
    #    csv_writer.writerow([sed_table[0][0][i],sed_table[0][1][i],sed_table[0][2][i]])

    #csv_file.close()
    ################################
    
    config = read_config('config.yaml')
    irfs = get_irfs(config)
    livetime = u.Quantity(config['pointing']['livetime']).to('second')
    
    filename_on = 'npred_cube_convolved.fits'
    
    cube_on = SkyCube.read(filename_on)
    on_pos = SkyCoord(83.6333 * u.deg, 22.0144 * u.deg, frame='icrs')
    on_size = 0.50 * u.deg #0.167
    
    off_pos = SkyCoord(83.6333 * u.deg, 22.0144 * u.deg, frame='icrs')
    off_size = 0.50 * u.deg #0.167
    
    on_region = CircleSkyRegion(on_pos, on_size)
    off_region = CircleSkyRegion(off_pos, off_size)
    #mask = cube_on.region_mask(on_region)
    
    ## data=cube_on.spectrum(on_region)['value'].data
    
    on_vector = PHACountsSpectrum(energy_lo = cube_on.energies('edges')[:-1],energy_hi= cube_on.energies('edges')[1:],data=cube_on.spectrum(on_region)['value'].data * u.ct, meta={'EXPOSURE' : livetime.value})  # cube_sed(cube_on, mask=mask)[0][1] * u.ct

    filename_off = 'noff_cube_convolved.fits'
    
    cube_off = SkyCube.read(filename_off)
    #mask = cube_off.region_mask(on_region)


    aeff = EffectiveAreaTable.from_parametrization(energy = cube_on.spectrum(on_region)['e_ref'],instrument = 'CTA')   #.data * u.Unit('TeV')
    edisp = EnergyDispersion.from_gauss(e_true = aeff.energy.bins , e_reco = on_vector.energy.bins)

    
    off_vector = PHACountsSpectrum(energy_lo = cube_off.energies('edges')[:-1],energy_hi= cube_off.energies('edges')[1:],data= cube_off.spectrum(off_region) ['value'].data * u.ct, meta={'EXPOSURE' : livetime.value, 'OFFSET' : 0.3 * u.deg})


    sed_table = SpectrumObservation(on_vector = on_vector, off_vector = off_vector, aeff = aeff, edisp = edisp)
    #print sed_table.alpha
    #print sed_table.livetime
    sed_table.peek()
    plt.show()

    
    model2fit1 = LogParabola(amplitude=1e-4 * u.Unit('cm-2 s-1 TeV-1') , reference=1 * u.TeV, alpha=2.5 * u.Unit('')  , beta=0.1 * u.Unit('')  )
    model2fit2 = ExponentialCutoffPowerLaw(index = 1. * u.Unit(''), amplitude = 1e-4 * u.Unit('cm-2 s-1 TeV-1'), reference= 1 * u.TeV ,  lambda_= 0. * u.Unit('TeV-1') )
    model2fit3 = PowerLaw(index= 2.0 * u.Unit(''), amplitude= 5e-4 * u.Unit('cm-2 s-1 TeV-1') , reference= 1 * u.TeV)

    models2fit = [model2fit1,model2fit2,model2fit3]


    for k in range(len(models2fit)):
        fit_source = SpectrumFit(obs_list = sed_table, model=models2fit[k],forward_folded=True, fit_range=(0.5 * u.Unit('TeV'), 50 * u.Unit('TeV')) )
        fit_source.fit()
        fit_source.est_errors()
        results = fit_source.result
        ax0, ax1 = results[0].plot(figsize=(8,8))
        #plt.show()
        print(results[0])


if __name__ == '__main__':
    main()
