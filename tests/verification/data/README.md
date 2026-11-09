# Reference data

ECSV tables with units; read with `astropy.table.QTable.read(path, format="ascii.ecsv")`. Missing values are NaN or
masked. Citations live in `docs/source/docs_bib.bib`.

- `sn2013dg/photometry.ecsv` — AB griz photometry of the superluminous supernova SN 2013dg (z = 0.265), transcribed from
  the table of the discovery paper, `Nicholl_2014` (arXiv:1405.1325, from its TeX source, so the band of every cell is
  unambiguous). The magnitudes are as published: not corrected for Galactic extinction (`ebv_mw` in the metadata).
  Two limiting magnitudes are flagged `upper_limit`, and four measurements that the authors say may include host light
  are flagged `host_contaminated`.
- `sn2013dg/published_fit.ecsv` — the SN 2013dg row of the magnetar fits of `Nicholl_2017` (arXiv:1706.00825): median
  with upper and lower 1-sigma errors, in the units of that table.
