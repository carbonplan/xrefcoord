import pytest


@pytest.mark.parametrize(
    "ex_path",
    [
        "https://github.com/fsspec/kerchunk/blob/main/kerchunk/tests/lcmap_tiny_cog_2019.tif?raw=true",
        "s3://sentinel-cogs/sentinel-s2-l2a-cogs/44/P/MT/2022/5/S2A_44PMT_20220525_0_L2A/AOT.tif",
    ],
)
def test_end_2_end(ex_path):
    import fsspec
    import xarray as xr
    from kerchunk.tiff import tiff_to_zarr

    # load xarray/rioxarray data
    rio_ds = xr.open_dataset(ex_path, engine="rasterio")

    # load kerchunk reference dataset
    out = tiff_to_zarr(ex_path)
    gcm = fsspec.get_mapper("reference://", fo=out)
    ref_ds = xr.open_dataset(gcm, engine="zarr", backend_kwargs={"consolidated": False})

    ds = ref_ds.xref.generate_coords(x_dim_name="X", y_dim_name="Y")

    # check that coords are identical as rioxarray generated coords from tiff (requires rioxarray)
    assert (rio_ds.x == ds["x"]).all()
    assert (rio_ds.y == ds["y"]).all()
