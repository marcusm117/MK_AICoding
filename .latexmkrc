$pdf_mode = 5;  # Use xelatex to generate PDF
$pdflatex = 'xelatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
$biber = 'biber %O %S';
