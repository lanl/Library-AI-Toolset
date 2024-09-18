### DINOCR (Digital INformation with Optical Character Regonition) ###

A tool for extracting data from non-searchable PDFs (but can also be used for searchable files). Capability exists for extraction of titles, headers, text bodies, figures and tables, and Latex (both block and in-line).

The *full_content_extract.py* file contains the extract() function, which is the main function used in information extraction. It takes a path to the PDF to be operated on, as well as a path to an output directory for the resulting markdown and image files.
