"""
PDF -> Markdown conversion via the `marker` library.

Import of `marker` (and its model loading) is deferred until a PDF is
actually converted, so the rest of the ingest pipeline doesn't pay the
startup cost, or require the dependency, when there are no PDFs to process.
"""

from pathlib import Path

_converter = None


def _get_converter():
    global _converter
    if _converter is None:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        _converter = PdfConverter(artifact_dict=create_model_dict())
    return _converter


def convert_pdf_to_markdown(path: Path) -> str:
    from marker.output import text_from_rendered

    rendered = _get_converter()(str(path))
    text, _, _ = text_from_rendered(rendered)
    return text
