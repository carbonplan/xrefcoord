import xarray as xr


def _assign_rename_coords(ds: xr.Dataset, coord_dict: dict, x_dim_name: str, y_dim_name: str):
    ds[x_dim_name] = coord_dict["x"]
    ds[y_dim_name] = coord_dict["y"]
    return ds.rename({x_dim_name: "x", y_dim_name: "y"})


def _drop_dims(
    ds: xr.Dataset, time_dim_name: str = None, x_dim_name: str = None, y_dim_name: str = None
) -> xr.Dataset:
    # Drops all dims except for named x, y and time dims
    dim_dict = {val for val in [time_dim_name, x_dim_name, y_dim_name] if val}
    return ds.drop_dims(list(set(ds.dims) - dim_dict))


def _get_shape(ds: xr.Dataset) -> tuple:
    return ds[list(ds.variables)[0]].shape  # 0th level


def _generate_multiscale_coords(ds: xr.Dataset) -> dict:
    """If reference TIFF dataset contains multiscales, ie pyramid levels, generate coords for each level and assign each coord to a leaf of a datatree object."""

    import datatree

    if "multiscales" not in ds.attrs:
        raise AttributeError(
            "multiscales missing from ds.attrs. Perhaps this reference tiff is only single level. If so, set `multiscales=False`"
        )

    levels = ds.attrs["multiscales"][0]["datasets"]

    multiscale_level_dict = {str(list(lvl.values())[0]): ds[list(lvl.values())] for lvl in levels}

    dt = datatree.DataTree.from_dict(multiscale_level_dict)

    for node in list(dt):
        dim_names = list(dt[node].ds.dims)

        dt[node].ds = dt[node].ds.xref.generate_ds_coords(
            time_dim_name=dim_names[0], y_dim_name=dim_names[1], x_dim_name=dim_names[2]
        )

    return dt


def _generate_coords(ds, x_dim_name: str, y_dim_name: str) -> dict:
    # Adapted from kerchunk

    import imagecodecs.numcodecs

    imagecodecs.numcodecs.register_codecs(verbose=False)

    if "ModelTiepoint" not in ds.attrs:
        raise AttributeError(
            "ModelTiepoint attribute missing from dataset attrs. Coordinate generation is not supported if this attribute is missing."
        )
    if "ModelPixelScale" not in ds.attrs:
        raise AttributeError(
            "ModelPixelScale attribute missing from dataset attrs. Coordinate generation is not supported if this attribute is missing."
        )
    import dask.array as da

    def gen_xcoords(ds):
        shape = ds[list(ds.variables)[0]].shape
        xscale = ds.attrs["ModelPixelScale"][0]
        x0 = ds.attrs["ModelTiepoint"][3]

        return xr.DataArray(da.arange(shape[-1]) * xscale + x0 + xscale / 2, dims="x")

    def gen_ycoords(ds):
        shape = ds[list(ds.variables)[0]].shape
        yscale = ds.attrs["ModelPixelScale"][1]
        y0 = ds.attrs["ModelTiepoint"][4]

        return xr.DataArray(da.arange(shape[-2]) * -yscale + y0 - yscale / 2, dims="y")

    ds[x_dim_name] = gen_xcoords(ds)
    ds[y_dim_name] = gen_ycoords(ds)

    return ds
