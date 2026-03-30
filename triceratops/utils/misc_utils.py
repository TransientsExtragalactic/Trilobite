"""Miscellaneous utility functions for Triceratops."""


# =========================================== #
# Unit Management Utilities                   #
# =========================================== #
def ensure_unit(value, unit):
    """
    Ensure that a value has the specified unit.

    Parameters
    ----------
    value : astropy.units.Quantity, int, or float
        The value to check or convert.
    unit : astropy.units.Unit
        The desired unit.

    Returns
    -------
    astropy.units.Quantity
        The value with the specified unit.
    """
    import astropy.units as u

    if isinstance(value, u.Quantity):
        return value.to(unit)
    else:
        return value * unit


def ensure_in_units(value, unit):
    """
    Ensure that ``value`` is converted to ``unit`` and return the value.

    Parameters
    ----------
    value : astropy.units.Quantity
        The value to check or convert.
    unit : astropy.units.Unit
        The desired unit.

    Returns
    -------
    float
        The value in the specified units.
    """
    import astropy.units as u

    if isinstance(value, u.Quantity):
        try:
            return value.to_value(unit)
        except u.UnitConversionError as err:
            raise ValueError(f"Cannot convert {value} to units of {unit}.") from err
    else:
        return value
