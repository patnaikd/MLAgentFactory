"""Test script to verify processing_complete message flow."""
import requests
import time
import json

API_BASE_URL = "http://localhost:8000"

def test_processing_complete_flow():
    """Test the complete flow: create session -> send query -> poll for completion."""
    print("🧪 Testing processing_complete message flow...\n")

    # Step 1: Create a session
    print("1️⃣ Creating new session...")
    response = requests.post(f"{API_BASE_URL}/sessions", json={"metadata": {"test": "processing_complete"}})
    response.raise_for_status()
    session_data = response.json()
    session_id = session_data["session_id"]
    print(f"✅ Session created: {session_id}\n")

    # Step 2: Send a simple query
    print("2️⃣ Sending query to agent...")
    query = "What is 2 + 2? Please answer briefly."
    response = requests.post(
        f"{API_BASE_URL}/sessions/{session_id}/query",
        json={"message": query}
    )
    response.raise_for_status()
    print(f"✅ Query sent: {query}\n")

    # Step 3: Poll for messages and check for processing_complete
    print("3️⃣ Polling for messages...\n")
    last_message_id = 0
    max_polls = 60  # Maximum 60 polls (30 seconds with 0.5s interval)
    poll_count = 0
    found_processing_complete = False

    while poll_count < max_polls:
        poll_count += 1

        # Get new messages
        response = requests.get(
            f"{API_BASE_URL}/sessions/{session_id}/messages",
            params={"since_message_id": last_message_id, "limit": 100}
        )
        response.raise_for_status()
        result = response.json()

        messages = result.get("messages", [])

        if messages:
            print(f"📨 Poll #{poll_count}: Received {len(messages)} new messages")

            for msg in messages:
                message_id = msg["message_id"]
                message_type = msg["message_type"]
                content = msg["content"]

                print(f"   - Message {message_id}: type={message_type}")

                # Check for processing_complete
                if message_type == "processing_complete":
                    print(f"\n✅ Found processing_complete message!")
                    print(f"   Content: {content}")
                    found_processing_complete = True

                # Update cursor
                last_message_id = message_id

            print()
        else:
            print(f"📭 Poll #{poll_count}: No new messages")

        # Check if we found the completion signal
        if found_processing_complete:
            print("🎉 SUCCESS: processing_complete message detected!")
            print("\nFlow verification:")
            print("✅ Session created")
            print("✅ Query sent")
            print("✅ Messages polled")
            print("✅ processing_complete signal received")
            print("\n✨ The UI should now stop polling and enable text input!")
            break

        # Wait before next poll
        time.sleep(0.5)

    if not found_processing_complete:
        print("❌ TIMEOUT: processing_complete message not received within 30 seconds")
        print(f"Total messages received: {last_message_id}")

    # Step 4: Get session stats
    print("\n4️⃣ Final session statistics:")
    response = requests.get(f"{API_BASE_URL}/sessions/{session_id}/stats")
    response.raise_for_status()
    stats = response.json()
    print(f"   - Status: {stats['status']}")
    print(f"   - Total messages: {stats['message_count']}")
    print(f"   - Total cost: ${stats['total_cost']:.4f}")
    print(f"   - Process alive: {stats['process_alive']}")

    return found_processing_complete

if __name__ == "__main__":
    try:
        success = test_processing_complete_flow()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
