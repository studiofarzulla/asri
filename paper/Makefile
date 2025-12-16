# ============================================================================
# ASRI Paper Build Makefile
# ============================================================================
# Usage:
#   make          - Build PDF (4-pass for proper citations)
#   make clean    - Remove build artifacts
#   make watch    - Continuous rebuild on file changes (requires fswatch)
#   make check    - Validate BibTeX entries
# ============================================================================

PAPER = ASRI_Paper
LATEX = pdflatex
BIBTEX = bibtex

.PHONY: all clean watch check

# Default target: full 4-pass build
all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib
	@echo "=== Pass 1: Initial LaTeX compilation ==="
	$(LATEX) $(PAPER).tex
	@echo "=== Pass 2: BibTeX for citations ==="
	$(BIBTEX) $(PAPER)
	@echo "=== Pass 3: Resolve citations ==="
	$(LATEX) $(PAPER).tex
	@echo "=== Pass 4: Final cross-references ==="
	$(LATEX) $(PAPER).tex
	@echo ""
	@echo "=== Build complete: $(PAPER).pdf ==="

# Quick build (single pass, no bibliography)
quick:
	$(LATEX) $(PAPER).tex

# Clean build artifacts
clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log
	rm -f $(PAPER).out $(PAPER).toc $(PAPER).lof $(PAPER).lot
	rm -f $(PAPER).fls $(PAPER).fdb_latexmk $(PAPER).synctex.gz
	@echo "=== Cleaned build artifacts ==="

# Deep clean (includes PDF)
distclean: clean
	rm -f $(PAPER).pdf
	@echo "=== Removed PDF ==="

# Watch for changes and rebuild (requires fswatch)
watch:
	@echo "=== Watching for changes... (Ctrl+C to stop) ==="
	@while true; do \
		fswatch -1 $(PAPER).tex references.bib && make all; \
	done

# Check BibTeX for common issues
check:
	@echo "=== Checking BibTeX entries ==="
	@grep -n "title = {" references.bib | head -20
	@echo ""
	@echo "Total entries: $$(grep -c "^@" references.bib)"
	@echo ""
	@echo "Checking for missing DOIs..."
	@grep -B5 "year = {" references.bib | grep -v "doi" | grep "@" || echo "All entries have DOIs"

# Count words in main text (approximate)
wordcount:
	@detex $(PAPER).tex | wc -w
	@echo "words (approximate)"

# Generate diff against previous version (requires latexdiff)
diff:
	@if [ -f $(PAPER)_v1.tex ]; then \
		latexdiff $(PAPER)_v1.tex $(PAPER).tex > $(PAPER)_diff.tex; \
		$(LATEX) $(PAPER)_diff.tex; \
		echo "=== Diff generated: $(PAPER)_diff.pdf ==="; \
	else \
		echo "No previous version found. Save current as $(PAPER)_v1.tex first."; \
	fi
