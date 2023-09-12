# Download ERA5 data #

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


### Relevant citations ###
Hersbach, H., Bell, B., Berrisford, P., Hirahara, S., Horányi, A., Muñoz Sabater, J., Nicolas, J., Peubey, C., Radu, R., Schepers, D., Simmons, A., Soci, C., Abdalla, S., Abellan, X., Balsamo, G., Bechtold, P., Biavati, G., Bidlot, J., Bonavita, M., De Chiara, G., Dahlgren, P., Dee, D., Diamantakis, M., Dragani, R., Flemming, J., Forbes, R., Fuentes, M., Geer, A., Haimberger, L., Healy, S., Hogan, R. J., Hólm, E., Janisková, M., Keeley, S., Laloyaux, P., Lopez, P., Lupu, C., Radnoti, G., de Rosnay, P., Rozum, I., Vamborg, F., Villaume, S., and Thépaut, J.-N.: **The ERA5 global reanalysis**, Q. J. Royal Meteorol. Soc., 146, 1999–2049, https://doi.org/10.1002/qj.3803, 2020.

Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horányi, A., Muñoz Sabater, J., Nicolas, J., Peubey, C., Radu, R., Rozum, I., Schepers, D., Simmons, A., Soci, C., Dee, D., and Thépaut, J.-N.: **ERA5 monthly averaged data on single levels from 1940 to present.**, https://doi.org/10.24381/cds.f17050d7, 2023

Petzold, A., Neis, P., Rütimann, M., Rohs, S., Berkes, F., Smit, H. G. J., Krämer, M., Spelten, N., Spichtinger, P., Nédélec, P., and Wahner, A.: **Ice-supersaturated air masses in the northern mid-latitudes from regular in situ observations by passenger aircraft: vertical distribution, seasonality and tropospheric fingerprint**, Atmos. Chem. Phys., 20, 8157–8179, https://doi.org/10.5194/acp-20-8157-2020, 2020.

