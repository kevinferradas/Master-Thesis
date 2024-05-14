import numpy as np
from numba import njit


@njit
def float_string(x, decimals=1):
    assert decimals > 0
    
    integer_part = int(x)
    decimal_digits = np.empty((decimals,), dtype=np.int8)
    for i in range(decimals+1):
        x *= 10
        if i < decimals:
            decimal_digits[i] = int(abs(x) % 10)
        else:
            if int(abs(x) % 10) >= 5:
                decimal_digits[-1] += 1
    
    for j in range(decimals-1,0,-1):
        if decimal_digits[j] == 10:
            decimal_digits[j-1] += 1
            decimal_digits[j] = 0
        else:
            break
    
    if decimal_digits[0] == 10:
        integer_part += 1
        decimal_digits[0] = 0
    
    s = str(integer_part) + '.'
    for k in range(decimals):
        s += str(decimal_digits[k])
    return s


@njit
def exp_string(x, decimals=1):
    assert decimals > 0
    
    if x==0:
        s = "0."
        for k in range(decimals):
            s += "0"
        s += "e+000"
        return s
    
    exp = int(np.log10(abs(x)))
    if exp >= 0:
        while int(np.log10(abs(x))) != 0:
            x /= 10
    else:
        while int(np.log10(abs(x))) != 0:
            x *= 10
        x *= 10
    s = float_string(x, decimals)
    if abs(exp)//10 == 0:
        str_exp = "00"
    elif abs(exp)//10 != 0 and abs(exp)//100==0:
        str_exp = "0"
    else:
        str_exp = ""
    if exp >= 0:
        s += "e+" + str_exp + str(exp)
    else:
        s += "e-" + str_exp + str(abs(exp-1))
    
    return s
