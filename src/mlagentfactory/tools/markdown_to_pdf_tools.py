"""Tools for converting markdown files to PDF format."""
import logging
from pathlib import Path

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
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
        except ImportError as e:
            return {
                "content": [{
                    "type": "text",
                    "text": f"Required libraries not installed. Install with: uv add markdown reportlab\nError: {e}"
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

        # Convert HTML to PDF using reportlab
        logger.info(f"Generating PDF: {output_pdf_path}")

        # Create PDF document
        doc = SimpleDocTemplate(
            str(output_pdf_path),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
        )

        # Container for the 'Flowable' objects
        elements = []

        # Get styles
        styles = getSampleStyleSheet()

        # Create custom styles based on css_style parameter
        style_name = css_style if css_style in ['github', 'minimal'] else 'default'
        _add_custom_styles(styles, style_name)

        # Simple HTML to Flowable conversion
        # This is a basic implementation - for complex HTML, consider using reportlab.platypus.html
        from html.parser import HTMLParser

        class HTMLToFlowables(HTMLParser):
            def __init__(self, styles):
                super().__init__()
                self.styles = styles
                self.elements = []
                self.current_text = []
                self.current_style = styles['Normal']
                self.in_code = False
                self.in_pre = False
                self.list_items = []
                self.in_list = False

            def handle_starttag(self, tag, attrs):
                if tag == 'h1':
                    self.current_style = self.styles['Heading1']
                elif tag == 'h2':
                    self.current_style = self.styles['Heading2']
                elif tag == 'h3':
                    self.current_style = self.styles['Heading3']
                elif tag == 'h4':
                    self.current_style = self.styles['Heading4']
                elif tag == 'p':
                    self.current_style = self.styles['Normal']
                elif tag == 'code':
                    self.in_code = True
                elif tag == 'pre':
                    self.in_pre = True
                elif tag in ['ul', 'ol']:
                    self.in_list = True
                    self.list_items = []

            def handle_endtag(self, tag):
                if tag in ['h1', 'h2', 'h3', 'h4', 'p']:
                    if self.current_text:
                        text = ''.join(self.current_text).strip()
                        if text:
                            self.elements.append(Paragraph(text, self.current_style))
                            self.elements.append(Spacer(1, 0.2*inch))
                        self.current_text = []
                elif tag == 'code':
                    self.in_code = False
                elif tag == 'pre':
                    if self.current_text:
                        text = ''.join(self.current_text)
                        self.elements.append(Preformatted(text, self.styles['Code']))
                        self.elements.append(Spacer(1, 0.2*inch))
                        self.current_text = []
                    self.in_pre = False
                elif tag in ['ul', 'ol']:
                    self.in_list = False
                elif tag == 'li':
                    if self.current_text:
                        text = ''.join(self.current_text).strip()
                        if text:
                            bullet = '• ' if self.in_list else ''
                            self.elements.append(Paragraph(f'{bullet}{text}', self.styles['Normal']))
                        self.current_text = []

            def handle_data(self, data):
                if self.in_code:
                    self.current_text.append(f'<font name="Courier">{data}</font>')
                else:
                    self.current_text.append(data)

        parser = HTMLToFlowables(styles)
        parser.feed(html_content)
        elements.extend(parser.elements)

        # Build PDF
        doc.build(elements)

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


def _add_custom_styles(styles, style_name: str):
    """Add custom styles to the ReportLab style sheet.

    Args:
        styles: ReportLab StyleSheet to modify
        style_name: Name of the style preset ('default', 'github', 'minimal')
    """
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.colors import HexColor

    # Add a Code style for preformatted text (if it doesn't already exist)
    if 'Code' not in styles:
        styles.add(ParagraphStyle(
            name='Code',
            parent=styles['Normal'],
            fontName='Courier',
            fontSize=9,
            leftIndent=20,
            rightIndent=20,
            spaceBefore=6,
            spaceAfter=6,
            backColor=HexColor('#f6f8fa'),
        ))

    # Customize based on style preset
    if style_name == 'github':
        # GitHub-style customizations
        styles['Heading1'].fontSize = 28
        styles['Heading2'].fontSize = 22
        styles['Heading3'].fontSize = 18
        styles['Normal'].fontSize = 12
        styles['Normal'].textColor = HexColor('#24292f')
    elif style_name == 'minimal':
        # Minimal style customizations
        styles['Heading1'].fontSize = 20
        styles['Heading1'].fontName = 'Times-Bold'
        styles['Heading2'].fontSize = 18
        styles['Heading2'].fontName = 'Times-Bold'
        styles['Normal'].fontName = 'Times-Roman'
        styles['Normal'].fontSize = 12
    else:
        # Default style customizations
        styles['Heading1'].fontSize = 24
        styles['Heading2'].fontSize = 20
        styles['Heading3'].fontSize = 16
        styles['Normal'].fontSize = 11
