import pytest

from xrefcoord.validation import _validate_attrs


@pytest.mark.parametrize("attrs", [{"ModelPixelScale": "1", "ModelTiepoint": "test"}])
def test_validate_attrs(attrs):
    _validate_attrs(attrs)


# tests exception logic for missing both required attrs or missing single required attrs
@pytest.mark.parametrize("attrs", [{"none": "none"}, {"ModelTiepoint": "test"}])
def test_validate_attrs_fail(attrs):
    with pytest.raises(AttributeError):
        _validate_attrs(attrs)
