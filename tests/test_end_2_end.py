

import pytest



def test_end_2_end():

    import xarray as xr
    import fsspec
    import imagecodecs.numcodecs
    imagecodecs.numcodecs.register_codecs()
    from kerchunk.tiff import tiff_to_zarr, generate_coords
    import xrefcoord
    # load xarray/rioxarray data

    ex_path = 'https://github.com/fsspec/kerchunk/blob/main/kerchunk/tests/lcmap_tiny_cog_2019.tif?raw=true'
    rio_ds = xr.open_dataset(ex_path,engine='rasterio')

    # load kerchunk reference dataset
    out = tiff_to_zarr(ex_path)
    gcm = fsspec.get_mapper("reference://", fo=out)
    ref_ds = xr.open_dataset(
    gcm, engine="zarr", backend_kwargs={"consolidated": False})

    ds = ref_ds.xref.generate_coords(x_dim_name='X',y_dim_name='Y')
    

    # check that coords are identical as rioxarray generated coords from tiff (requires rioxarray)
    assert (rio_ds.x == ds['x']).all()
    assert (rio_ds.y == ds['y']).all()


