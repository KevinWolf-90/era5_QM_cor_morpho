import numpy as np

def eSatLiquid(T):
    a = -6096.4642   #--polynomes for the water vapor saturation pressure for t below freezing
    b = 16.635794    #--after Sonntag, 1994
    c = 2.7245552E-2 #--https://www.eas.ualberta.ca/jdwilson/EAS372_13/Vomel_CIRES_satvpformulae.html
    d = 1.6853396E-5
    f = 2.4575506
    
    if (isinstance(T, float)) or (isinstance(T, int)):
        if T < 0:
            T = 0
    else:
        T[T < 0] = 0
    
    e_sat_liq = np.exp(a / T + b - c*T + d*T**2 + f*np.log(T))
    e_sat_liq = e_sat_liq * 100 # conversion to Pa
    return e_sat_liq
    
def eSatIce(T):
    a = 9.550426   #--polynomes for the water vapor saturation pressure
    b = 5723.265   #--after Murphy and Koop, 2005
    c = 3.53068    #--https://www.eas.ualberta.ca/jdwilson/EAS372_13/Vomel_CIRES_satvpformulae.html
    d = 0.00728332
    
    if (isinstance(T, float)) or (isinstance(T, int)):
        if T < 0:
            T = 0
    else:
        T[T < 0] = 0
            
    e_sat_ice = np.exp(a - b / T + c * np.log(T) - d * T) 
    return e_sat_ice
