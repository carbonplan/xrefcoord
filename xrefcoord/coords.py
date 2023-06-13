import xarray as xr


def _assign_rename_coords(ds: xr.Dataset, coord_dict: dict, x_dim_name: str, y_dim_name: str):
    ds[x_dim_name] = coord_dict["x"]
    ds[y_dim_name] = coord_dict["y"]
    return ds.rename({x_dim_name: "x", y_dim_name: "y"})


def _drop_dims(ds: xr.Dataset, x_dim_name: str, y_dim_name: str) -> xr.Dataset:
    # Drops all dims except for named x, y dims
    return ds.drop_dims(list(set(ds.dims) - {x_dim_name, y_dim_name}))


def _get_shape(ds: xr.Dataset) -> tuple:
    return ds[list(ds.variables)[0]].shape  # 0th level


def _generate_coords(attrs: dict, shape: tuple) -> dict:
    # Taken from kerchunk
    """Produce coordinate arrays for given variable

    Specific to GeoTIFF input attributes

    Parameters
    ----------
    attrs: dict
        Containing the geoTIFF tags, probably the root group of the dataset
    shape: tuple[int]
        The array size in numpy (C) order
    """
    import imagecodecs.numcodecs
    import numpy as np

    imagecodecs.numcodecs.register_codecs()

    height, width = shape[-2:]
    xscale, yscale, zscale = attrs["ModelPixelScale"][:3]
    x0, y0, z0 = attrs["ModelTiepoint"][3:6]
    out = {}
    out["x"] = np.arange(width) * xscale + x0 + xscale / 2
    out["y"] = np.arange(height) * -yscale + y0 - yscale / 2
    if len(shape) > 2:
        out["z"] = np.arange(shape[-3]) * zscale + z0 + zscale / 2
    return out
