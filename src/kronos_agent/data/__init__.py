"""Candle data: stores, archives, decision windows, the chronological partition (step 2).

Nothing is imported here on purpose: the command line imports ``data.timeutil`` and
``data.partition`` (standard library only) at ``--help`` time, and every other module of the
package pulls in numpy and pandas. ``docs/DATA.md`` is the design of this package.
"""
