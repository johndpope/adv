(TeX-add-style-hook
 "report"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "11pt" "twoside" "a4paper")))
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art11"
    "amssymb"
    "amsmath"
    "graphicx"
    "float"))
 :latex)

