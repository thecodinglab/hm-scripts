# Calculates the maximum floating point number based on the given base, number of digits in the mantissa and
# the maximum exponent
#
# Formula: (base ^ e_max) - (base ^ (e_max - n))
def x_max(base: int, n: int, e_max: int) -> float:
    return base ** e_max * (1 - base ** (-n))


# Calculates the smallest positive floating point number based on the given base and the minimum exponent
#
# Formula: base ^ (e_min - 1)
def x_min(base, e_min):
    return base ** (e_min - 1)


# Calculates the largest relative error when rounding a floating point number to a given number of digits
def eps(base, n):
    return base ** (1 - n)
