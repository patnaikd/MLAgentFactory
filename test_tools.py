"""
Tests for MLAgentFactory tools.
"""

import asyncio
import tempfile
from pathlib import Path

from tools import write_file


async def test_write_file():
    """Test the write_file tool with a temporary file."""

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "test_output.txt"
        test_content = "Hello from test!"

        print(f"Testing write_file tool...")
        print(f"Temporary file path: {tmp_path}")

        # Call the write_file tool handler
        result = await write_file.handler({
            "path": str(tmp_path),
            "content": test_content
        })

        # Check the result
        print(f"\nTool result: {result}")

        # Verify the file was created
        assert tmp_path.exists(), "File was not created"
        print("✓ File exists")

        # Verify the content
        actual_content = tmp_path.read_text()
        assert actual_content == test_content, f"Content mismatch: expected '{test_content}', got '{actual_content}'"
        print(f"✓ Content matches: '{actual_content}'")

        # Verify the result message
        assert "content" in result, "Result missing 'content' key"
        assert len(result["content"]) > 0, "Result content is empty"
        assert result["content"][0]["type"] == "text", "Result content type is not 'text'"
        assert "Successfully wrote" in result["content"][0]["text"], "Success message not found in result"
        print(f"✓ Result message correct: {result['content'][0]['text']}")

        print("\n✅ All tests passed!")
        print(f"Temporary directory will be cleaned up automatically")


async def test_write_file_error():
    """Test the write_file tool with an invalid path."""

    print("\n" + "=" * 50)
    print("Testing write_file error handling...")

    # Try to write to an invalid path (directory that doesn't exist)
    invalid_path = "/invalid/path/that/does/not/exist/test.txt"

    result = await write_file.handler({
        "path": invalid_path,
        "content": "This should fail"
    })

    print(f"Tool result: {result}")

    # Verify error handling
    assert "is_error" in result and result["is_error"], "Expected error flag"
    assert "Error writing file" in result["content"][0]["text"], "Error message not found"
    print(f"✓ Error handled correctly: {result['content'][0]['text']}")

    print("\n✅ Error handling test passed!")


async def main():
    """Run all tests."""
    print("Running write_file tool tests...")
    print("=" * 50)

    await test_write_file()
    await test_write_file_error()

    print("\n" + "=" * 50)
    print("🎉 All tests completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
