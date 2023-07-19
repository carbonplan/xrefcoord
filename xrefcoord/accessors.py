import datatree
import xarray as xr

from .coords import (
    _assign_rename_coords,
    _drop_dims,
    _generate_coords,
    _generate_multiscale_coords,
    _get_shape,
)
from .validation import _validate_attrs


class XRefAccessor:
    """
    Dataset accessor functionality.
    """

    def __init__(self, xarray_obj: xr.Dataset):
        self.xarray_obj = xarray_obj


@xr.register_dataset_accessor("xref")
class XRefDatasetAccessor(XRefAccessor):
    """xrefcoord accessor for xarray dataset objects"""

    def validate_attrs(self):
        _validate_attrs(self.xarray_obj.attrs)

    def generate_multiscale_coords(self) -> datatree.DataTree:
        """If a reference TIFF contains multiscales (pyramids), generates coords for each pyramid level
        and assigns each pyramid level to the leaf of a datatree.


        """
        dt = _generate_multiscale_coords(ds=self.xarray_obj)
        return dt

    def generate_ds_coords(
        self, time_dim_name: str = None, x_dim_name: str = None, y_dim_name: str = None
    ) -> xr.Dataset:
        """Generate coords
        :param time_dim_name: Time dimension name to keep
        : param time_dim_name: str
        :param x_dim_name: X dimension name to keep
        :type x_dim_name: str
        :param y_dim_name: Y dimension name to keep
        :type y_dim_name: str
        """
        # Validate
        self.validate_attrs()

        # Generate shape
        shape = _get_shape(self.xarray_obj)

        # Generate coords
        coord_dict = _generate_coords(self.xarray_obj.attrs, shape)

        # Drop extra multiscale dims
        ds = _drop_dims(
            ds=self.xarray_obj,
            time_dim_name=time_dim_name,
            x_dim_name=x_dim_name,
            y_dim_name=y_dim_name,
        )

        # Assign and rename coords
        ds = _assign_rename_coords(
            ds=ds, coord_dict=coord_dict, x_dim_name=x_dim_name, y_dim_name=y_dim_name
        )
        return ds
