# STATUS — A->B report PDF + CSV rewritten for IRC:82-2023 (started 2026-11-08)

Owner files: app/pci_report_pdf.py, app/pci_treatments_2023.py (new), report_csv() in app/pci_routes.py,
scripts/tests/test_pci_pdf.py, CSV tests in scripts/tests/test_pci_routes.py, C:\GeoAI\pci\preview_2023\*.
Not touched: app/pci_core.py, app/pci_chainage.py, app/pci_irc2023.py (report JSON = input).

## Decisions
- Headline = summary.pci (length-weighted mean of km rows). Headline parameter table: values = whole-stretch
  quantities; PCI_i = length-weighted mean of the rated km rows' PCI_i (so sum w_i PCI_i == headline).
- Map / section colour: single PCI -> Table 5.5 class (6 colours). No single PCI -> worst Tables 5.1-5.3 range
  ("Excellent to Good" / "Satisfactory to Fair" / "Poor to Fail") among the parameters that have a value,
  labelled as NOT the PCI. White "could be worse" dashes dropped.
- Treatments: 2023 clause numbers verified in C:\GeoAI\pci\irc82_2023\IRC82-2023.txt (app/pci_treatments_2023.py).
  Table 5.5 "Renewal (as per section 9.5)": no §9.5 in the 2023 text; periodic renewal is §10 (Table 10.1).

## Steps
1. [done, iterating] Rewrite app/pci_report_pdf.py (+ app/pci_treatments_2023.py). Old 2015 module backed up
   in the session scratchpad (pci_report_pdf_2015.py); git has it too.
2. [done] CSV report_csv(): '#' header lines + one row per 100 m / km / stretch, 2023 columns.
3. [done] Tests: test_pci_pdf.py 12/12; test_pci_routes.py CSV tests pass (test_report_mdr_category fails in
   pci_core.class_treatment: _RENEWAL has no 'mdr' key - not my file, reported)
4. [done 2026-11-08] Preview PDFs (script: scratchpad pv_db.py, builds on geoai_test, saves nothing) C:\GeoAI\pci\preview_2023\ from geoai_test, inspect every page

## Result (2026-11-08)
- 8 sample inputs rendered from geoai_test (nothing saved): C:\GeoAI\pci\preview_2023\<rule>.pdf/.csv/.json,
  0 overlap/off-page problems; pages viewed. Single-PCI state: preview_2023/tests/test_calibrated.pdf.
- Open: pci_core.class_treatment KeyError 'mdr' (2015 _RENEWAL) breaks test_report_mdr_category (not my file).
  Annex selection rule is still the 2015 one in pci_chainage._annex (pothole runs + sections whose 2015 band
  can be Poor); the PDF words it neutrally.
