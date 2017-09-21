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
from astropy.coordinates import SkyCoord, Angle
from gammapy.cube import make_exposure_cube
from gammapy.cube.utils import SkyCube    # compute_npred_cube compute_npred_cube_simple
from gammapy.image import SkyImage
from gammapy.irf import EffectiveAreaTable2D, EnergyDispersion2D
from gammapy.irf import EnergyDependentMultiGaussPSF
from configuration import get_model_gammapy, make_ref_cube
from astropy.io import fits
from gammapy.background import FOVCube
from gammapy.utils.energy import EnergyBounds

def _validate_inputs(flux_cube, exposure_cube):
    if flux_cube.data.shape[1:] != exposure_cube.data.shape[1:]:
        raise ValueError('flux_cube and exposure cube must have the same shape!\n'
                         'flux_cube: {0}\nexposure_cube: {1}'
                         ''.format(flux_cube.data.shape[1:], exposure_cube.data.shape[1:]))

def compute_npred_cube(flux_cube, exposure_cube, ebounds,
                       integral_resolution=10):
    """Compute predicted counts cube.
        
        Parameters
        ----------
        flux_cube : `SkyCube`
        Flux cube, really differential surface brightness in 'cm-2 s-1 TeV-1 sr-1'
        exposure_cube : `SkyCube`
        Exposure cube
        ebounds : `~astropy.units.Quantity`
        Energy bounds for the output cube
        integral_resolution : int (optional)
        Number of integration steps in energy bin when computing integral flux.
        
        Returns
        -------
        npred_cube : `SkyCube`
        Predicted counts cube with energy bounds as given by the input ``ebounds``.
        
        See also
        --------
        compute_npred_cube_simple
        
        Examples
        --------
        Load an example dataset::
        
        from gammapy.datasets import FermiGalacticCenter
        from gammapy.utils.energy import EnergyBounds
        from gammapy.irf import EnergyDependentTablePSF
        from gammapy.cube import SkyCube, compute_npred_cube
        
        filenames = FermiGalacticCenter.filenames()
        flux_cube = SkyCube.read(filenames['diffuse_model'], format='fermi-background')
        exposure_cube = SkyCube.read(filenames['exposure_cube'], format='fermi-exposure')
        psf = EnergyDependentTablePSF.read(filenames['psf'])
        
        Compute an ``npred`` cube and a PSF-convolved version::
        
        flux_cube = flux_cube.reproject(exposure_cube)
        ebounds = EnergyBounds([10, 30, 100, 500], 'GeV')
        npred_cube = compute_npred_cube(flux_cube, exposure_cube, ebounds)
        
        kernels = psf.kernels(npred_cube)
        npred_cube_convolved = npred_cube.convolve(kernels)
        """
    _validate_inputs(flux_cube, exposure_cube)
    
    # Make an empty cube with the requested energy binning
    sky_geom = exposure_cube.sky_image_ref
    energies = EnergyBounds(ebounds)
    npred_cube = SkyCube.empty_like(sky_geom, energies=energies, unit='', fill=np.nan)
    
    # Process and fill one energy bin at a time
    for idx in range(len(ebounds) - 1):
        emin, emax = ebounds[idx: idx + 2]
        ecenter = np.sqrt(emin * emax)
        
        flux = flux_cube.sky_image_integral(emin, emax, interpolation='linear', nbins=integral_resolution)
        
        
        exposure = exposure_cube.sky_image(ecenter, interpolation='linear')
        solid_angle = exposure.solid_angle()
        # to remove the nans we put 0! not clear if correct though
        flux.data[np.isnan(flux.data)]=0
        exposure.data[np.isnan(exposure.data)]=0
        npred = flux.data * exposure.data * solid_angle
        
        #print solid_angle
        #print flux.data
        #print exposure.data
        #print npred
        
        npred_cube.data[idx] = npred.value
    
    return npred_cube

def compute_npred_cube_simple(flux_cube, exposure_cube):
    """Compute predicted counts cube (using a simple method).
        
        * Simply multiplies flux and exposure and pixel solid angle and energy bin width.
        * No spatial reprojection, or interpolation or integration in energy.
        * This is very fast, but can be inaccurate (e.g. for very large energy bins.)
        * If you want a more fancy method, call `compute_npred_cube` instead.
        
        Output cube energy bounds will be the same as for the exposure cube.
        
        Parameters
        ----------
        flux_cube : `SkyCube`
        Differential flux cube
        exposure_cube : `SkyCube`
        Exposure cube
        
        Returns
        -------
        npred_cube : `SkyCube`
        Predicted counts cube
        
        See also
        --------
        compute_npred_cube
        """
    _validate_inputs(flux_cube, exposure_cube)
    
    solid_angle = exposure_cube.sky_image_ref.solid_angle()
    # TODO: is this OK? Exposure cube has no `ebounds`, only `ecenter`, but npred needs `ebounds`, no?
    de = exposure_cube.energy_width
    flux = flux_cube.data
    exposure = exposure_cube.data
    npred = flux * exposure * solid_angle * de[:, np.newaxis, np.newaxis]
    
    npred_cube = SkyCube.empty_like(exposure_cube)
    npred_cube.data = npred.value
    return npred_cube


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


def compute_spatial_model_integral(model, image):
    """
    This is just used for debugging here.
    TODO: remove or put somewhere into Gammapy as a utility function or method?
    """
    coords = image.coordinates()
    surface_brightness = model(coords.data.lon.deg, coords.data.lat.deg) * u.Unit('deg-2')
    solid_angle = image.solid_angle()
    return (surface_brightness * solid_angle).sum().to('')


def read_config(filename):
    with open(filename) as fh:
        config = yaml.load(fh)

    # TODO: fix the following issue in a better way (e.g. raise an error or fix somehow)
    # apparently this gets returned as string, but we want float!?
    # prefactor1: 2e-12
    config['model']['prefactor'] = float(config['model']['prefactor'])

    return config

def compute_nexcess_cube(npred_cube,bkg_rate,livetime):
    print "In the function"
    
    ebin = npred_cube.energies(mode="edges")
    ebounds = EnergyBounds(ebin)
    #    livetime = u.Quantity(config['pointing']['livetime']).to('second')
    
    #    sky_geom = npred_cube.sky_image_ref
    #    nexcess_cube = SkyCube.empty_like(sky_geom, energies=ebounds, unit='TeV', fill=np.nan)
    config = read_config('config.yaml')
    nexcess_cube = make_ref_cube(config)
    
    #nexcess_cube = SkyCube.empty(npred_cube)
    # For each energy bin, I need to obtain the correct background rate (two, one for the on and one for the off)
    for idx in range(len(ebounds) - 1):
        emin, emax = ebounds[idx: idx + 2]
        ecenter = np.sqrt(emin * emax)
        print emin, emax
        
        npred = npred_cube.sky_image_idx(idx)
        #        npred = npred_cube.sky_image_integral(emin, emax, interpolation='linear')
        npred.unit = u.Unit('TeV')
        solid_angle = npred.solid_angle()
        npred.data[np.isnan(npred.data)]=0.
        
        ## First I obtain the corresponding energy bin in the background cube
        energy_bin = bkg_rate.energy_edges.find_energy_bin(ecenter)
        ## Then I get the corresponding offset angle (keep in mind for very extended sources that will be incorrect - we need to get a dependency with the offset)
        coord_bin = bkg_rate.find_coord_bin(coord=Angle([0., 0.], 'deg'))
        ## I obtain the background rate in that particular energy bin
        ## I scale for the livetime (what do I do with the solid angle? I need to correct when I produce the new cube)
        bkg_rate_ebin = bkg_rate.data[energy_bin, coord_bin[1], coord_bin[0]] * livetime *solid_angle * (emax - emin).to("MeV")
        ## I create gaussian variations of my background rate

        bkg_gauss1 = np.random.normal(bkg_rate_ebin,np.sqrt(bkg_rate_ebin))
        bkg_gauss2 = np.random.normal(bkg_rate_ebin,np.sqrt(bkg_rate_ebin))
        
        nbkg1_ebin = SkyImage.empty_like(npred, unit='TeV', fill=bkg_gauss1)
        nbkg2_ebin = SkyImage.empty_like(npred, unit='TeV', fill=bkg_gauss2)
        ## DEBUG
        print npred
        print nbkg1_ebin
        print nbkg2_ebin
        nexcess = npred.data + nbkg1_ebin.data - nbkg2_ebin.data
        
        #print nexcess_cube.data[idx]
        nexcess_cube.data[idx] = nexcess    #.to('TeV')
    
    return nexcess_cube



def compute_sum_cube(flux_cube, flux_cube2):
    ebin = flux_cube.energies(mode="edges")
    ebounds = EnergyBounds(ebin)
    
    config = read_config('config.yaml')
    nexcess_cube_sum = make_ref_cube(config)
    #nexcess_cube_sum = SkyCube.empty(flux_cube)
    # For each energy bin, I need to obtain the correct background rate (two, one for the on and one for the off)
    for idx in range(len(ebounds) - 1):
        npred1 = flux_cube.sky_image_idx(idx)
        npred2 =flux_cube2.sky_image_idx(idx)
        #npred = npred_cube.sky_image_integral(emin, emax, interpolation='linear')

        ## DEBUG
        print npred1.data
        print npred2.data

        nexcess_sum = u.Quantity(npred1.data.value + npred2.data.value,'1 / (cm2 s sr TeV)')
        #print nexcess_sum.value
        #print nexcess_cube_sum.data[idx]
        nexcess_cube_sum.data[idx] = nexcess_sum.value
        
    return nexcess_cube_sum



def compute_npredoff_cube(npred_cube,bkg_rate,livetime):
    print "In the function"
    
    ebin = npred_cube.energies(mode="edges")
    ebounds = EnergyBounds(ebin)
    #    livetime = u.Quantity(config['pointing']['livetime']).to('second')
    
    #    sky_geom = npred_cube.sky_image_ref
    config = read_config('config.yaml')
    noff_cube = make_ref_cube(config)
    
    # For each energy bin, I need to obtain the correct background rate (two, one for the on and one for the off)
    for idx in range(len(ebounds) - 1):
        emin, emax = ebounds[idx: idx + 2]
        ecenter = np.sqrt(emin * emax)
        
        npred = npred_cube.sky_image_idx(idx)
        #        npred = npred_cube.sky_image_integral(emin, emax, interpolation='linear')
        npred.unit = u.Unit('TeV')
        solid_angle = npred.solid_angle()
        npred.data[np.isnan(npred.data)]=0.
        
        ## First I obtain the corresponding energy bin in the background cube
        energy_bin = bkg_rate.energy_edges.find_energy_bin(ecenter)
        ## Then I get the corresponding offset angle (keep in mind for very extended sources that will be incorrect - we need to get a dependency with the offset)
        coord_bin = bkg_rate.find_coord_bin(coord=Angle([0., 0.], 'deg'))
        ## I obtain the background rate in that particular energy bin
        ## I scale for the livetime (what do I do with the solid angle? I need to correct when I produce the new cube)
        bkg_rate_ebin = bkg_rate.data[energy_bin, coord_bin[1], coord_bin[0]] * livetime *solid_angle * (emax - emin).to("MeV")
        ## I create gaussian variations of my background rate
        bkg_gauss1 = np.random.normal(bkg_rate_ebin,np.sqrt(bkg_rate_ebin))
        bkg_gauss2 = np.random.normal(bkg_rate_ebin,np.sqrt(bkg_rate_ebin))
        
        nbkg1_ebin = SkyImage.empty_like(npred, unit='TeV', fill=bkg_gauss1)
        nbkg2_ebin = SkyImage.empty_like(npred, unit='TeV', fill=bkg_gauss2)
        ## DEBUG
        n_off = nbkg1_ebin.data   #   - nbkg2_ebin.data
        
        #print nexcess_cube.data[idx]
        noff_cube.data[idx] = n_off    #.to('TeV')
    
    return noff_cube





def main():
    config = read_config('config.yaml')
    config2 = read_config('config2.yaml')
    # getting the IRFs, effective area and PSF
    irfs = get_irfs(config)
    # create an empty reference image
    ref_cube = make_ref_cube(config)
    if config['binning']['coordsys'] == 'CEL':
        pointing = SkyCoord(config['pointing']['ra'], config['pointing']['dec'], frame='icrs', unit='deg')
    if config['binning']['coordsys'] == 'GAL':
        pointing = SkyCoord(config['pointing']['glat'], config['pointing']['glon'], frame='galactic', unit='deg')

    #ref_cube2 = SkyCube.empty(ref_cube)
    ref_cube2 = make_ref_cube(config2)
    livetime = u.Quantity(config['pointing']['livetime']).to('second')
    exposure_cube = make_exposure_cube(
        pointing=pointing,
        livetime=livetime,
        aeff=irfs['aeff'],
        ref_cube=ref_cube,
        offset_max=Angle(config['selection']['ROI']),
    )
    print('exposure sum: {}'.format(np.nansum(exposure_cube.data)))
    exposure_cube.data = exposure_cube.data.to('m2 s')
    print(exposure_cube)

    # Define model and do some quick checks
    model = get_model_gammapy(config)
    model2 = get_model_gammapy(config2)

    spatial_integral = compute_spatial_model_integral(model.spatial_model, exposure_cube.sky_image_ref)
    spatial_integral2 = compute_spatial_model_integral(model2.spatial_model, exposure_cube.sky_image_ref)

    print('Spatial integral (should be 1): ', spatial_integral)
    flux_integral = model.spectral_model.integral(emin='1 TeV', emax='10 TeV')
    print('Integral flux in range 1 to 10 TeV: ', flux_integral.to('cm-2 s-1'))
    flux_integral2 = model2.spectral_model.integral(emin='1 TeV', emax='10 TeV')



# import IPython; IPython.embed()

    # Compute PSF-convolved npred in a few steps
    # 1. flux cube
    # 2. npred_cube
    # 3. apply PSF
    # 4. apply EDISP

    flux_cube = model.evaluate_cube(ref_cube)
    flux_cube2 = model2.evaluate_cube(ref_cube2)

    flux_sum=compute_sum_cube(flux_cube,flux_cube2)
    
    from time import time
    t0 = time()
    npred_cube = compute_npred_cube(
        flux_sum, exposure_cube,
        ebounds=flux_sum.energies('edges'),
        integral_resolution=2,
    )
    bkg = irfs['bkg']
    #    nexcess_cube = compute_nexcess_cube(
    #        npred_cube,bkg,livetime)


    t1 = time()
    npred_cube_simple = compute_npred_cube_simple(flux_sum, exposure_cube)

    t2 = time()
    print('npred_cube: ', t1 - t0)
    print('npred_cube_simple: ', t2 - t1)
    print('npred_cube sum: {}'.format(np.nansum(npred_cube.data)))     #.to('').data)))
    print('npred_cube_simple sum: {}'.format(np.nansum(npred_cube_simple.data)))   #.to('').data)))

    nexcess_cube = compute_nexcess_cube(
        npred_cube,bkg,livetime)



    noff_cube = compute_npredoff_cube(npred_cube,bkg,livetime)


    # Apply PSF convolution here
    #    kernels = irfs['psf'].kernels(npred_cube_simple)
    #    npred_cube_convolved = npred_cube_simple.convolve(kernels)

    kernels = irfs['psf'].kernels(nexcess_cube)
    npred_cube_convolved = nexcess_cube.convolve(kernels)

    noff_cube_convolved = noff_cube.convolve(kernels)
    # TODO: apply EDISP here!

    # Compute counts as a Poisson fluctuation
    # counts_cube = SkyCube.empty_like(ref_cube)
    # counts_cube.data = np.random.poisson(npred_cube_convolved.data)

    # Debugging output

    print('npred_cube sum: {}'.format(np.nansum(npred_cube.data.to('').data)))
    print('npred_cube_convolved sum: {}'.format(np.nansum(npred_cube_convolved.data.to('').data)))

    #    print('nexcess_cube sum: {}'.format(np.nansum(nexcess_cube.data.to('').data)))
    #    print('nexcess_cube_convolved sum: {}'.format(np.nansum(nexcess_cube_convolved.data.to('').data)))
    # TODO: check that sum after PSF convolution or applying EDISP are the same



    exposure_cube.write('exposure_cube.fits', overwrite=True, format='fermi-exposure')
    flux_sum.write('flux_cube.fits.gz', overwrite=True)
    nexcess_cube.write('npred_cube.fits.gz', overwrite=True)
    npred_cube_convolved.write('npred_cube_convolved.fits.gz', overwrite=True)
    noff_cube.write('noff_cube.fits.gz', overwrite=True)
    noff_cube_convolved.write('noff_cube_convolved.fits.gz', overwrite=True)
    # npred_cube2 = SkyCube.read('npred_cube.fits.gz')
    # print(npred_cube2)


if __name__ == '__main__':
    main()
