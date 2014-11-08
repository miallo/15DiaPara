all: Prohaupt.tex
	pdflatex Prohaupt.tex
	bibtex Prohaupt
	pdflatex Prohaupt.tex
	pdflatex Prohaupt.tex
	rm -f *.toc *.log *.aux *.bbl *.blg

clean: 
	rm -f $(PROG) *.o *.toc *.log *.aux *.bbl *.blg
