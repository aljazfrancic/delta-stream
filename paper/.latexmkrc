# latexmk configuration for paper.tex.
#
# The svg package converts ../figures/*.svg via Inkscape at build time, which
# requires TeX shell-escape; adding it here keeps the documented build command
# a plain `latexmk -pdf paper.tex`.
$pdf_mode = 1;                                    # pdflatex route
set_tex_cmds('-shell-escape -interaction=nonstopmode %O %S');
$bibtex_use = 2;                                  # run bibtex; clean .bbl on -C
$clean_ext .= ' %R.synctex.gz';
# Note: the svg package's Inkscape exports land in ./svg-inkscape/ — a cache,
# git-ignored; delete the directory by hand if you want a truly cold rebuild.
