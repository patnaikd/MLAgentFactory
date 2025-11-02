"""Tools for converting markdown files to PDF format."""
import logging
from pathlib import Path
from typing import Optional

from claude_agent_sdk import tool

logger = logging.getLogger(__name__)


@tool(
    "markdown_to_pdf",
    "Convert a markdown file to PDF format with optional styling (supports 'default', 'github', 'minimal' styles or custom CSS file path)",
    {
        "markdown_file": str,
        "output_pdf": {"type": str, "optional": True},
        "css_style": {"type": str, "optional": True}
    }
)
async def markdown_to_pdf(args: dict) -> dict:
    """Convert a markdown file to PDF format.

    This tool reads a markdown file, converts it to HTML, and generates a PDF with proper styling.
    Supports standard markdown syntax including headers, lists, code blocks, tables, and images.

    Args:
        args: Dictionary containing:
            - markdown_file: Path to the markdown file to convert
            - output_pdf: (Optional) Path for the output PDF file (if not provided, uses same name as markdown_file with .pdf extension)
            - css_style: (Optional) CSS style preset ('default', 'github', 'minimal') or path to custom CSS file

    Returns:
        Dictionary with content blocks for the tool result

    Examples:
        # Convert with default styling
        markdown_to_pdf({"markdown_file": "README.md"})

        # Convert with custom output path
        markdown_to_pdf({"markdown_file": "docs/guide.md", "output_pdf": "output/guide.pdf"})

        # Use GitHub-style formatting
        markdown_to_pdf({"markdown_file": "README.md", "css_style": "github"})

        # Use custom CSS file
        markdown_to_pdf({"markdown_file": "README.md", "css_style": "styles/custom.css"})
    """
    try:
        # Extract parameters from args
        markdown_file = args.get("markdown_file")
        output_pdf = args.get("output_pdf")
        css_style = args.get("css_style")
        # Import required libraries
        try:
            import markdown
            from weasyprint import HTML, CSS
            from weasyprint.text.fonts import FontConfiguration
        except ImportError as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Required libraries not installed. Install with: uv add markdown weasyprint\nError: {e}"
                }],
                "is_error": True
            }

        # Validate input file
        md_path = Path(markdown_file)
        if not md_path.exists():
            return {
                "content": [{
                    "type": "text",
                    "text": f"Markdown file not found: {markdown_file}"
                }],
                "is_error": True
            }

        if not md_path.is_file():
            return {
                "content": [{
                    "type": "text",
                    "text": f"Path is not a file: {markdown_file}"
                }],
                "is_error": True
            }

        # Determine output path
        if output_pdf is None:
            output_pdf_path = md_path.with_suffix('.pdf')
        else:
            output_pdf_path = Path(output_pdf)
            # Create parent directories if they don't exist
            output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

        # Read markdown content
        logger.info(f"Reading markdown file: {markdown_file}")
        markdown_content = md_path.read_text(encoding='utf-8')

        # Convert markdown to HTML
        logger.info("Converting markdown to HTML")
        md_processor = markdown.Markdown(
            extensions=[
                'extra',           # Tables, fenced code blocks, etc.
                'codehilite',      # Syntax highlighting for code blocks
                'toc',             # Table of contents
                'nl2br',           # Newline to <br>
                'sane_lists'       # Better list handling
            ]
        )
        html_content = md_processor.convert(markdown_content)

        # Get CSS styling
        css_content = _get_css_style(css_style)

        # Create full HTML document
        full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{md_path.stem}</title>
    <style>
        {css_content}
    </style>
</head>
<body>
    <div class="container">
        {html_content}
    </div>
</body>
</html>
"""

        # Convert HTML to PDF
        logger.info(f"Generating PDF: {output_pdf_path}")
        font_config = FontConfiguration()
        html_doc = HTML(string=full_html)

        # Use CSS for styling
        stylesheets = [CSS(string=css_content, font_config=font_config)]
        html_doc.write_pdf(
            output_pdf_path,
            stylesheets=stylesheets,
            font_config=font_config
        )

        logger.info(f"Successfully created PDF: {output_pdf_path}")

        return {
            "content": [{
                "type": "text",
                "text": f"Successfully converted {markdown_file} to {output_pdf_path.absolute()}"
            }]
        }

    except Exception as e:
        logger.error(f"Error converting markdown to PDF: {e}", exc_info=True)
        return {
            "content": [{
                "type": "text",
                "text": f"Failed to convert markdown to PDF: {str(e)}"
            }],
            "is_error": True
        }


def _get_css_style(css_style: Optional[str]) -> str:
    """Get CSS styling based on the specified style preset or custom file.

    Args:
        css_style: Style preset name ('default', 'github', 'minimal') or path to CSS file

    Returns:
        CSS content as string
    """
    # If no style specified, use default
    if css_style is None:
        css_style = 'default'

    # Check if it's a file path
    css_path = Path(css_style)
    if css_path.exists() and css_path.is_file():
        logger.info(f"Using custom CSS file: {css_style}")
        return css_path.read_text(encoding='utf-8')

    # Use preset styles
    if css_style == 'github':
        return _get_github_style()
    elif css_style == 'minimal':
        return _get_minimal_style()
    else:
        return _get_default_style()


def _get_default_style() -> str:
    """Get default CSS styling for PDF output."""
    return """
        @page {
            size: letter;
            margin: 1in;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #333;
        }

        .container {
            max-width: 100%;
        }

        h1, h2, h3, h4, h5, h6 {
            margin-top: 24pt;
            margin-bottom: 12pt;
            font-weight: 600;
            line-height: 1.25;
            page-break-after: avoid;
        }

        h1 {
            font-size: 24pt;
            border-bottom: 1px solid #eee;
            padding-bottom: 8pt;
        }

        h2 {
            font-size: 20pt;
            border-bottom: 1px solid #eee;
            padding-bottom: 8pt;
        }

        h3 { font-size: 16pt; }
        h4 { font-size: 14pt; }
        h5 { font-size: 12pt; }
        h6 { font-size: 11pt; }

        p {
            margin-top: 0;
            margin-bottom: 12pt;
        }

        a {
            color: #0366d6;
            text-decoration: none;
        }

        code {
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
            font-size: 9pt;
            background-color: #f6f8fa;
            padding: 2pt 4pt;
            border-radius: 3pt;
        }

        pre {
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
            font-size: 9pt;
            background-color: #f6f8fa;
            padding: 12pt;
            border-radius: 6pt;
            overflow-x: auto;
            margin-bottom: 12pt;
            page-break-inside: avoid;
        }

        pre code {
            background-color: transparent;
            padding: 0;
        }

        blockquote {
            margin: 0;
            padding-left: 12pt;
            border-left: 3pt solid #dfe2e5;
            color: #6a737d;
        }

        table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 12pt;
            page-break-inside: avoid;
        }

        table th,
        table td {
            padding: 6pt 12pt;
            border: 1pt solid #dfe2e5;
        }

        table th {
            background-color: #f6f8fa;
            font-weight: 600;
        }

        table tr:nth-child(even) {
            background-color: #f6f8fa;
        }

        ul, ol {
            margin-top: 0;
            margin-bottom: 12pt;
            padding-left: 24pt;
        }

        li {
            margin-bottom: 4pt;
        }

        img {
            max-width: 100%;
            height: auto;
        }

        hr {
            height: 0;
            border: 0;
            border-top: 1pt solid #dfe2e5;
            margin: 24pt 0;
        }

        /* Code highlighting */
        .codehilite {
            background-color: #f6f8fa;
            padding: 12pt;
            border-radius: 6pt;
            margin-bottom: 12pt;
            page-break-inside: avoid;
        }
    """


def _get_github_style() -> str:
    """Get GitHub-flavored CSS styling for PDF output."""
    return """
        @page {
            size: letter;
            margin: 0.75in;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif;
            font-size: 12pt;
            line-height: 1.6;
            color: #24292f;
            background-color: white;
        }

        .container {
            max-width: 100%;
        }

        h1, h2, h3, h4, h5, h6 {
            margin-top: 24pt;
            margin-bottom: 16pt;
            font-weight: 600;
            line-height: 1.25;
            page-break-after: avoid;
        }

        h1 {
            font-size: 28pt;
            padding-bottom: 0.3em;
            border-bottom: 1px solid #d0d7de;
        }

        h2 {
            font-size: 22pt;
            padding-bottom: 0.3em;
            border-bottom: 1px solid #d0d7de;
        }

        h3 { font-size: 18pt; }
        h4 { font-size: 14pt; }
        h5 { font-size: 12pt; }
        h6 { font-size: 11pt; color: #57606a; }

        p {
            margin-top: 0;
            margin-bottom: 16pt;
        }

        a {
            color: #0969da;
            text-decoration: none;
        }

        code {
            font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
            font-size: 10pt;
            background-color: rgba(175, 184, 193, 0.2);
            padding: 2pt 4pt;
            border-radius: 6pt;
        }

        pre {
            font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
            font-size: 10pt;
            background-color: #f6f8fa;
            padding: 16pt;
            border-radius: 6pt;
            overflow-x: auto;
            margin-bottom: 16pt;
            page-break-inside: avoid;
        }

        pre code {
            background-color: transparent;
            padding: 0;
        }

        blockquote {
            margin: 0;
            padding-left: 16pt;
            border-left: 4pt solid #d0d7de;
            color: #57606a;
        }

        table {
            border-collapse: collapse;
            border-spacing: 0;
            width: 100%;
            margin-bottom: 16pt;
            page-break-inside: avoid;
        }

        table th,
        table td {
            padding: 6pt 13pt;
            border: 1pt solid #d0d7de;
        }

        table th {
            background-color: #f6f8fa;
            font-weight: 600;
        }

        table tr {
            background-color: white;
            border-top: 1pt solid #d0d7de;
        }

        table tr:nth-child(2n) {
            background-color: #f6f8fa;
        }

        ul, ol {
            margin-top: 0;
            margin-bottom: 16pt;
            padding-left: 32pt;
        }

        li + li {
            margin-top: 4pt;
        }

        img {
            max-width: 100%;
            height: auto;
        }

        hr {
            height: 2pt;
            padding: 0;
            margin: 24pt 0;
            background-color: #d0d7de;
            border: 0;
        }

        .codehilite {
            background-color: #f6f8fa;
            padding: 16pt;
            border-radius: 6pt;
            margin-bottom: 16pt;
            page-break-inside: avoid;
        }
    """


def _get_minimal_style() -> str:
    """Get minimal CSS styling for PDF output."""
    return """
        @page {
            size: letter;
            margin: 1in;
        }

        body {
            font-family: "Times New Roman", Times, serif;
            font-size: 12pt;
            line-height: 1.5;
            color: #000;
        }

        .container {
            max-width: 100%;
        }

        h1, h2, h3, h4, h5, h6 {
            margin-top: 18pt;
            margin-bottom: 12pt;
            font-weight: bold;
            page-break-after: avoid;
        }

        h1 { font-size: 20pt; }
        h2 { font-size: 18pt; }
        h3 { font-size: 16pt; }
        h4 { font-size: 14pt; }
        h5 { font-size: 12pt; }
        h6 { font-size: 12pt; }

        p {
            margin-top: 0;
            margin-bottom: 12pt;
        }

        a {
            color: #000;
            text-decoration: underline;
        }

        code, pre {
            font-family: "Courier New", Courier, monospace;
            font-size: 10pt;
        }

        pre {
            background-color: #f5f5f5;
            padding: 12pt;
            margin-bottom: 12pt;
            border: 1pt solid #ccc;
            page-break-inside: avoid;
        }

        blockquote {
            margin: 0;
            padding-left: 12pt;
            border-left: 2pt solid #ccc;
            font-style: italic;
        }

        table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 12pt;
            page-break-inside: avoid;
        }

        table th,
        table td {
            padding: 6pt;
            border: 1pt solid #000;
        }

        table th {
            font-weight: bold;
        }

        ul, ol {
            margin-top: 0;
            margin-bottom: 12pt;
            padding-left: 24pt;
        }

        img {
            max-width: 100%;
            height: auto;
        }

        hr {
            border: 0;
            border-top: 1pt solid #000;
            margin: 18pt 0;
        }
    """
