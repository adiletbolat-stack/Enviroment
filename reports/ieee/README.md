# IEEE Overleaf starter

This folder contains a modular IEEE conference-paper draft based on the
repository's study plan and the supplied Zotero Better BibTeX export.

## Files

- main.tex: IEEEtran conference entry point and package setup.
- sections/01_introduction.tex: funnel-style introduction and contributions.
- sections/02_methodology.tex: measures, workflow figure, analysis plan,
  limitations, and synthetic-data warning.
- figures/study_results.png: 300-DPI synthetic study-results graphic used
  in the Results section.
- references.bib: cleaned entries for cited sources from bb.bib.

The manuscript cites 10 peer-reviewed journal articles. Every bibliography
entry is cited in the text; no preprint-only records are included in the
submission bibliography.

## Overleaf

Upload this folder's files while preserving the sections/ directory, set
main.tex as the main document, and select pdfLaTeX. Overleaf runs BibTeX
for the IEEEtran bibliography style. Recompile until citations and section
references resolve. Review all bibliographic metadata against the source
publications before submission.

The document uses the balance package immediately before the bibliography to
balance the final two-column page. The checked-in main.pdf is the previous
four-page build; regenerate it after any source changes.

The supplied design PDF was not available to the current text-extraction
tools during drafting. This draft therefore uses the repository's README,
preregistration, configuration, and methodology notes. Confirm that the PDF
does not contain later decisions that supersede those files.
