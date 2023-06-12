
def _validate_attrs(attrs: dict):
    """ Verify that 'ModelPixelScale' & 'ModelTiepoint' exist in dataset attrs


    :param attrs: Dataset attributes
    :type attrs: dict
    """

    if not all(atr in attrs for atr in ("ModelPixelScale","ModelTiepoint")):
        raise AttributeError("'ModelPixelScale' and/or 'ModelTiepoint' does not exist in attrs.")
    
