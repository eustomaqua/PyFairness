# coding: utf-8
# pyfair.plain.


from pyfair.facil.utils_saver import elegant_print, get_elogger
from pyfair.facil.utils_timer import (
    fantasy_timer, fantasy_durat, elegant_durat, elegant_dated)

from pyfair.facil.ensem_voting import (
    plurality_voting, majority_voting, weighted_voting)


__all__ = [
    'weighted_voting',
    'plurality_voting',
    'majority_voting',

    'elegant_print',
    'fantasy_timer',
    'elegant_durat',
    'elegant_dated',
    'get_elogger',
]
