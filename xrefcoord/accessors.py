import xarray as xr

from .coords import _assign_rename_coords, _drop_dims, _generate_coords, _get_shape
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

    def generate_coords(self, x_dim_name: str, y_dim_name: str):
        """_summary_

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
        ds = _drop_dims(ds=self.xarray_obj, x_dim_name=x_dim_name, y_dim_name=y_dim_name)

        # Assign and rename coords
        ds = _assign_rename_coords(
            ds=ds, coord_dict=coord_dict, x_dim_name=x_dim_name, y_dim_name=y_dim_name
        )
        return ds
