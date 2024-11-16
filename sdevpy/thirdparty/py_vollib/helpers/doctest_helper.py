# -*- coding: utf-8 -*-

"""
py_vollib.helpers.doctest_helper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A library for option pricing, implied volatility, and
greek calculation.  py_vollib is based on lets_be_rational,
a Python wrapper for LetsBeRational by Peter Jaeckel as
described below.

:copyright: © 2017 Gammon Capital LLC
:license: MIT, see LICENSE for more details.

About LetsBeRational:
~~~~~~~~~~~~~~~~~~~~~

The source code of LetsBeRational resides at www.jaeckel.org/LetsBeRational.7z .

======================================================================================
Copyright © 2013-2014 Peter Jäckel.

Permission to use, copy, modify, and distribute this software is freely granted,
provided that this notice is preserved.

WARRANTY DISCLAIMER
The Software is provided "as is" without warranty of any kind, either express or implied,
including without limitation any implied warranties of condition, uninterrupted use,
merchantability, fitness for a particular purpose, or non-infringement.
======================================================================================
"""

# -----------------------------------------------------------------------------
# IMPORTS

# Standard library imports

# Related third party imports

# Local application/library specific imports


def run_doctest():
    hline = '**********************************************************************'
    import doctest
    result = doctest.testmod()
    if result.failed:
        print(hline)
        print('{} out of {} tests failed'.format(
            result.failed, result.attempted))
        print(hline)
    else:    
        print(hline)
        print('{} out of {} tests passed.'.format(
            result.attempted, result.attempted))
        print(hline)


if __name__ == "__main__":
    from py_vollib.helpers.doctest_helper import run_doctest
    run_doctest()
