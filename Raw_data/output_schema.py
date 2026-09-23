"""Species groups shared by the extraction scripts.

The order of these two lists defines the row/column order of every output
table. All other table dimensions (community names, replicate order, ...)
follow the order in which they appear in the raw workbooks.
"""

GROWERS = [
    'Ai', 'Ac', 'Bfi', 'Bf', 'Bt', 'Bu', 'Bx', 'Ba', 'Csp', 'Dl', 'Ls', 'Mf',
    'Pm', 'Bd', 'Bv', 'Rg',
]

NONGROWERS = [
    'Af', 'Ao', 'As', 'Bl.s', 'Col', 'Cs', 'Et', 'Eu.c', 'Eu.l', 'Im', 'Ld',
    'Lsp', 'Mi', 'Pc', 'Va', 'Vp',
]
