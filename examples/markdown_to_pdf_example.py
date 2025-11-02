"""Example demonstrating markdown to PDF conversion.

This example shows how to use the markdown_to_pdf tool both directly and through the ChatAgent.
"""
import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlagentfactory.tools.markdown_to_pdf_tools import markdown_to_pdf
from mlagentfactory.agents.chat_agent import ChatAgent


async def demo_direct_usage():
    """Demonstrate direct usage of the markdown_to_pdf tool."""
    print("=" * 60)
    print("DIRECT USAGE DEMO")
    print("=" * 60)

    # Create a sample markdown file
    sample_md_path = Path("./data/sample_document.md")
    sample_md_path.parent.mkdir(parents=True, exist_ok=True)

    sample_content = """# Sample Document

This is a demonstration of markdown to PDF conversion.

## Features

The markdown to PDF converter supports:

1. **Headers** of all levels (h1-h6)
2. **Text formatting** like *italic*, **bold**, and `inline code`
3. **Code blocks** with syntax highlighting
4. **Tables** for structured data
5. **Lists** (ordered and unordered)
6. **Blockquotes** for callouts
7. **Links** and images

## Code Example

Here's a Python code block:

```python
def hello_world():
    print("Hello, World!")
    return True

# Call the function
hello_world()
```

## Table Example

| Style    | Description                    | Best For           |
|----------|--------------------------------|--------------------|
| Default  | Clean, modern styling          | General documents  |
| GitHub   | GitHub-flavored appearance     | READMEs, docs      |
| Minimal  | Simple, classic style          | Academic papers    |

## Blockquote Example

> This is a blockquote. It's useful for highlighting important information
> or including quotes from other sources.

## List Example

**Supported Markdown Features:**
- Headers and subheaders
- Text formatting (bold, italic, code)
- Code blocks with highlighting
- Tables with alternating row colors
- Ordered and unordered lists
- Blockquotes
- Horizontal rules
- Links and images (when paths are valid)

---

## Conclusion

This demonstrates the full range of markdown features supported by the PDF converter.
The output should be well-formatted and ready for sharing or printing.
"""

    print(f"\n1. Creating sample markdown file: {sample_md_path}")
    sample_md_path.write_text(sample_content, encoding='utf-8')
    print(f"   ✓ Sample file created ({len(sample_content)} bytes)")

    # Test 1: Convert with default styling
    print("\n2. Converting with default styling...")
    result1 = await markdown_to_pdf({
        "markdown_file": str(sample_md_path),
        "output_pdf": "./data/sample_default.pdf"
    })
    if result1.get("is_error"):
        print(f"   ✗ Error: {result1['content'][0]['text']}")
    else:
        print(f"   ✓ {result1['content'][0]['text']}")

    # Test 2: Convert with GitHub styling
    print("\n3. Converting with GitHub styling...")
    result2 = await markdown_to_pdf({
        "markdown_file": str(sample_md_path),
        "output_pdf": "./data/sample_github.pdf",
        "css_style": "github"
    })
    if result2.get("is_error"):
        print(f"   ✗ Error: {result2['content'][0]['text']}")
    else:
        print(f"   ✓ {result2['content'][0]['text']}")

    # Test 3: Convert with minimal styling
    print("\n4. Converting with minimal styling...")
    result3 = await markdown_to_pdf({
        "markdown_file": str(sample_md_path),
        "output_pdf": "./data/sample_minimal.pdf",
        "css_style": "minimal"
    })
    if result3.get("is_error"):
        print(f"   ✗ Error: {result3['content'][0]['text']}")
    else:
        print(f"   ✓ {result3['content'][0]['text']}")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print("\nGenerated PDFs:")
    for pdf_file in Path("./data").glob("sample_*.pdf"):
        size_kb = pdf_file.stat().st_size / 1024
        print(f"  - {pdf_file.name} ({size_kb:.1f} KB)")

    return str(sample_md_path)


async def demo_agent_usage(markdown_file: str):
    """Demonstrate using markdown_to_pdf through the ChatAgent."""
    print("\n" + "=" * 60)
    print("AGENT USAGE DEMO")
    print("=" * 60)

    agent = ChatAgent()
    await agent.initialize()

    try:
        # Ask the agent to convert the markdown to PDF
        print(f"\nAsking agent to convert {markdown_file} to PDF...")
        query = f"Please convert the markdown file '{markdown_file}' to a PDF using the GitHub style. Save it as 'agent_output.pdf' in the data directory."

        response_parts = []
        async for chunk in agent.send_message(query):
            if chunk["type"] == "text":
                response_parts.append(chunk["content"])
                print(chunk["content"], end="", flush=True)
            elif chunk["type"] == "tool_use":
                print(f"\n  [Using tool: {chunk['tool_name']}]")
            elif chunk["type"] == "tool_result":
                if not chunk["is_error"]:
                    print(f"  [Tool completed successfully]")

        print("\n\n✓ Agent completed the task")
        print(f"  Total cost: ${agent.total_cost:.4f}")

    finally:
        await agent.cleanup()


async def main():
    """Run all demos."""
    print("Markdown to PDF Conversion Demo")
    print("=" * 60)

    # First, demonstrate direct usage
    markdown_file = await demo_direct_usage()

    # Then, demonstrate agent usage
    await demo_agent_usage(markdown_file)

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nCheck the ./data/ directory for generated PDF files.")


if __name__ == "__main__":
    asyncio.run(main())
