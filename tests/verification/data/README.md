# Reference data

ECSV tables with units; read with `astropy.table.QTable.read(path, format="ascii.ecsv")`. Missing values are masked.
Citations live in `docs/source/docs_bib.bib`.

- `sn2006aj/` — SN 2006aj / GRB 060218: `optical_photometry.ecsv` (UBVRI, host-subtracted and corrected for Galactic
  extinction, see `band_corrections.ecsv`), `object.ecsv` (redshift). Source: `Mirabal2006GRB060218` (attribution of the
  photometry itself unconfirmed).
- `prentice2016/arnett_fits.ecsv` — Arnett-model fits for 38 stripped-envelope supernovae, doi:10.1093/mnras/stv2983
  (BibTeX not yet added). `M_Ni`, `M_ej`, `E_K` carry `_err_up`/`_err_down`; `lit_*` are the literature values listed
  alongside, with `_lo`/`_hi` bounds; `lit_refs` are reference numbers in the paper.

Not yet known: Vega or AB magnitudes; t0 for SN 2006aj; cosmology; which Arnett formulation Prentice et al. used.
