# Documentation ERA5 and IAGOS analysis for paper #

This brief documentation accompanies the two manuscripts: 
1. "Correction of temperature and relative humidity biases in ERA5 by bivariate quantile mapping: Implications for contrail classification."
and
2. "Morphology and temporal and spatial distribution of non-persistent and persistent contrail formation areas in ERA5",
which are / will both be submitted to the journal Atmospheric Chemistry and Physics [(ACP)](https://www.atmospheric-chemistry-and-physics.net/)


## Download ERA5 data ##

ERA5 data of tempertaure, relative humidity, wind components, and cloud fraction were download from the [ERA5 data catalog (https://doi.org/10.24381/cds.f17050d7)](https://doi.org/10.24381/cds.f17050d7).
Due to the distribution of IAGOS flights, the focus of the analysis is on a domain between 30N and 70N, where most of the IGAOS observations are available. The latitudes were selected similar to Petzold et al 2020.

The data was downloaded with the following script: [ERA5_data_download.py](ERA5_data_download.py)

The code allows to specify:
<ul>
  <li> time steps (1, 3, or 6h) of the data </li>
  <li> years and months to be downloaded </li>
  <li> pressure levels </li>
  <li> area to download </li>
  <li> requested products </li>
</ul>



The comments are in the code and the code should be self-explaining. 

It is noted that the applied python interface and downloads from the ERA5 data base require a registration on the CDS homepage:
[https://cds.climate.copernicus.eu/cdsapp#!/home](https://cds.climate.copernicus.eu/cdsapp#!/home).
Please follow the instrctions given on the CDS homepage.

## Extraction of temperature, relative humidity,  wind components, and fraction of cloud cover from ERA5 ##
 
### General ###
ERA5 data is extracted along IAGOS flight trajectories. Flights from the years 2015 to 2021 are used. Extraction is performed with the nearest neighbor method by selecting the temporally and spatially closest ERA5 grid point.
The code that was used to extract data from ERA 5 with 1 hour resolution:
[A01_extract__along_flightpath_1h.py](A01_extract__along_flightpath_1h.py)

### Filtering ###
After the extraction the data is filtered. Only data is kept that fulfills the following criteria:

<ul>
  <li> IAGOS temperature value available  </li>
  <li> IAGOS temperature flagged for good or limited  </li>
  <li> Absolute difference between temperature from aircraft avionic system and IAGOS temperature is smaller than 5 K  </li>

  <li> IAGOS relative humidity (w.r.t. liquid water and ice) value available  </li>
  <li> Relative humidity w.r.t. ice flagged for good or limited  </li>
  <li> Relative humidity w.r.t. liquid water is between 0 and 100 %  </li>
  <li> Relative humidity w.r.t. ice is between 0 and 170 %  </li>
</ul>




### Smoothing ###
Two data sets are stored: 1) The original data that represents the native resolution of IAGOS with a measurement every 4 seconds. 2) A smoothed data set to better match the spatial resolution of IAGOS with the spatial resolution of ERA5; with however still a value every 4 seconds. The smoothing is realized by applying a Gaussian filter. The characteristics of the Gaussian filter are specified in Wolf et al 2023. The filter is not run over the data flags and only over the following variables:

<ul>
  <li> Temperature from IAGOS  </li>
  <li> Termpature from the aircraft avionic system  </li>
  <li> Particle number concentration  </li>
  <li> Relative humidity w.r.t. liquid water  </li>
  <li> Relative humidity w.r.t. ice  </li>
</ul>


### Storage ###

Yearly files of IAGOS and ERA5 along-track data is stored in a *.npz file that can be pickled.



## Quantile mapping ##

### Create cumulative distributions functions for quantile mapping ###

The inventory of cumulative distribution functions CDFs are created with:
[A02_create_cdf.py](A02_create_cdf.py)


### Application of quantile correction, general data analysis, and plots ###


The pickled ERA5, IAGOS data, and CDFs are read with: [A03_apply_2d_correction.py](A03_apply_2d_correction.py). This code also applies the quantile mapping corrections and includes the analysis that is presented in the manuscripts.

## Contrail formation climatology and morphology ##

### Create cumulative distributions functions for quantile mapping ###

Vertical distributions of contrail formation potential and vertical overlap were created with: [A06_era_3d.py](A06_era_3d.py)

Climatologies of temperature, relative humidity, windpseed, and persistent contrail formation were created with: [A07_monthly_crosssections.py](A07_monthly_crosssections.py)


# Relevant citations #
Hersbach, H., Bell, B., Berrisford, P., Hirahara, S., Horányi, A., Muñoz Sabater, J., Nicolas, J., Peubey, C., Radu, R., Schepers, D., Simmons, A., Soci, C., Abdalla, S., Abellan, X., Balsamo, G., Bechtold, P., Biavati, G., Bidlot, J., Bonavita, M., De Chiara, G., Dahlgren, P., Dee, D., Diamantakis, M., Dragani, R., Flemming, J., Forbes, R., Fuentes, M., Geer, A., Haimberger, L., Healy, S., Hogan, R. J., Hólm, E., Janisková, M., Keeley, S., Laloyaux, P., Lopez, P., Lupu, C., Radnoti, G., de Rosnay, P., Rozum, I., Vamborg, F., Villaume, S., and Thépaut, J.-N.: **The ERA5 global reanalysis**, Q. J. Royal Meteorol. Soc., 146, 1999–2049, [https://doi.org/10.1002/qj.3803](https://doi.org/10.1002/qj.3803), 2020.

Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horányi, A., Muñoz Sabater, J., Nicolas, J., Peubey, C., Radu, R., Rozum, I., Schepers, D., Simmons, A., Soci, C., Dee, D., and Thépaut, J.-N.: **ERA5 monthly averaged data on single levels from 1940 to present.**, [https://doi.org/10.24381/cds.f17050d7](https://doi.org/10.24381/cds.f17050d7), 2023

Petzold, A., Neis, P., Rütimann, M., Rohs, S., Berkes, F., Smit, H. G. J., Krämer, M., Spelten, N., Spichtinger, P., Nédélec, P., and Wahner, A.: **Ice-supersaturated air masses in the northern mid-latitudes from regular in situ observations by passenger aircraft: vertical distribution, seasonality and tropospheric fingerprint**, Atmos. Chem. Phys., 20, 8157–8179, [https://doi.org/10.5194/acp-20-8157-2020](https://doi.org/10.5194/acp-20-8157-2020), 2020.

Rap, A., Forster, P. M., Jones, A., Boucher, O., Haywood, J. M., Bellouin, N., and De Leon, R. R.: **Parameterization of contrails in the UK Met Office Climate Model**, J. Geophys. Res. Atmos., 115, [https://doi.org/https://doi.org/10.1029/2009JD012443](https://doi.org/https://doi.org/10.1029/2009JD012443), 2010.

Schumann, U.: **On conditions for contrail formation from aircraft exhausts**, Meteorologische Zeitschrift, 5, 1996.

Teoh, R., Schumann, U., Voigt, C., Schripp, T. Shapiro, M., Engberg, Z., Molloy, J., Koudis, G., and Stettler, M. E. J.: **Targeted Use of Sustainable Aviation Fuel to Maximize Climate Benefits**, Environ. Sci. Technol., [https://doi.org/10.1021/acs.est.2c05781](https://doi.org/10.1021/acs.est.2c05781), 2022b.

Wolf, K., Bellouin, N., and Boucher, O.: **Long-term upper-troposphere climatology of potential contrail occurrence over the Paris area derived from radiosonde observations**, Atmos. Chem. Phys., 23, 287–309, [https://doi.org/10.5194/acp-23-287-2023](https://doi.org/10.5194/acp-23-287-2023), 2023
