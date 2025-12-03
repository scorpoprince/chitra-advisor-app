"""
Symbol normalisation utilities.

These functions take user input and produce normalised ticker strings.  The
current implementation supports NSE and BSE symbols.  Any ticker without a
suffix will be assumed to be an NSE ticker.  Users may also specify BSE
tickers in the form ``BSE:<number>`` or ``<symbol>.BSE``.
"""

from .utils import logger


def normalize_symbol(text: str) -> str:
    """
    Normalise user input into a ticker string.

    Rules:
      * Trim spaces and uppercase the input.
      * If the input begins with ``BSE:``, strip the prefix and append ``.BSE``.
      * If the input already contains a period, assume the user provided a
        fully‑qualified ticker and return it unchanged.
      * Otherwise append ``.NS`` to assume an NSE listing.

    Examples:
      >>> normalize_symbol("TCS")
      'TCS.NS'

      >>> normalize_symbol("BSE:500325")
      '500325.BSE'

      >>> normalize_symbol("RELIANCE.BSE")
      'RELIANCE.BSE'
    """
    if not text or not text.strip():
        raise ValueError("Please enter a valid ticker or company name.")

    clean = text.strip().upper()

    # Accept fully qualified tickers like RELIANCE.BSE or RELIANCE.NS
    if "." in clean:
        return clean

    # BSE numeric form, e.g. BSE:500325
    if clean.startswith("BSE:"):
        return clean[4:] + ".BSE"

    # Default to NSE
    return clean + ".NS"


__all__ = ["normalize_symbol"]