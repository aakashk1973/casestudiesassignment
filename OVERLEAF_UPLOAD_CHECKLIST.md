# Overleaf upload checklist (Part 2 fixes)

## Replace / upload these files
1. Replace your Part 2 section with `04-deliberation.tex` contents
   (from `\section{Deliberation on Task 1}` through the appendix).
2. Upload regenerated figures (overwrite existing):
   - `image.png`  → Bank learning curve (train + val; duration excluded)
   - `image2.png` → Credit learning curve (train + val)
   - `image3.png` → Bank age Fairlearn (with n + base rate annotations)
   - `image4.png` → Credit sex Fairlearn (with n + base rate annotations)
3. Ensure `\usepackage{booktabs}` is in your preamble (tables use `\toprule`).
4. Add `bird2020fairlearn` from `references.bib` into `99-references.bib` if needed.

## Manual template cleanup in Overleaf (do this in your project settings / macros)
- Remove any `[[Remove before submitting...]]` placeholder text.
- Change running header from “Smith” to your name (Aakash K).
- Rename project / title from “Individual Task 2: Part 1” to the correct overall assessment title.
- Keep Condition 3 AI declaration + Appendix sections from `04-deliberation.tex`.

## What changed analytically
- Bank Marketing Part 2 **excludes `duration`**.
- Learning curves now plot **training and validation** PR-AUC.
- Fairlearn tables/figures include **subgroup n** and **base rates**.
- Numbers updated after re-run (bank DP≈0.57, EO≈0.52 without duration).
