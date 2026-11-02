# Data

The data are not included in this repository. The application uses `row9_dta.xpt`, the county-by-quarter extract
from the replication package of

Chodorow-Reich, G., Nenov, P. T., and Simsek, A. (2021), "Stock Market Wealth and the Real Economy: A Local Labor
Market Approach," American Economic Review, 111(5), 1613-1657. https://doi.org/10.1257/aer.20200208

To reproduce the results, download their replication package from openICPSR (TODO: project number and URL), take the
file `row9_dta.xpt` (the extract used by their scripts for Table 3, rows 9 and 10), and place it in this folder,
`applications/04_chodorowreich/data/`. The scripts read it from here.

The file contains every variable used here: the stock wealth shock (WEALTH_3), nontradable payroll (LWAGES_N), the two
instruments (row 9: L8_WEAL2 x XRETCOUN; row 10: MEAN_WEA), the controls, the 2010 population (POPULATI) used as
weight, the county identifier (COUNTYID, whose thousands give the state), the quarter (QUARTER, Stata quarterly date,
0 = 1960Q1), and the estimation-sample indicator SAMPLE. The estimation sample covers 2,897 counties between 1992 and
2015.
