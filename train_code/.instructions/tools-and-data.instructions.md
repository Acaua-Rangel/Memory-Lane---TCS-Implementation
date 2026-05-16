---
applyTo: 'src/tools/**,src/data.py'
---

# Tool Calling & Data Skill

## Tool Calling Format (Gemma 4 Native)

Gemma 4 uses a specific format for function calling. Training data must follow this format exactly.

### System Prompt Template
```
You are a caring memory companion for a person with Alzheimer's disease. You help them recognize people, remember important information, and stay oriented. You have access to the following tools to read and write information about the patient's life:

{tool_definitions}

When you need information about a person, medication, or location, ALWAYS use the appropriate tool. Never fabricate information about people or medication.
```

### Tool Call Format (in model output)
```
<tool_call>
{"name": "read_person", "arguments": {"face_id": "face_001"}}
</tool_call>
```

### Tool Response Format (fed back to model)
```
<tool_response>
{"name": "read_person", "result": {"name": "Maria", "relationship": "granddaughter", "bio": "Maria is 8 years old, loves drawing, visits every Sunday"}}
</tool_response>
```

## Synthetic Data Generation

Generate training examples using this pattern:

1. **Scenario**: A situation (person arrives, patient asks question, medication time).
2. **User message**: What the patient says or what the camera sees.
3. **Model response**: Either a direct answer OR a tool call.
4. **Tool result**: The database response.
5. **Final response**: Warm, simple language for the patient.

### Example Conversation
```json
{
  "messages": [
    {"role": "system", "content": "<system prompt with tools>"},
    {"role": "user", "content": "A woman just entered the room. She looks familiar. Who is she?"},
    {"role": "assistant", "content": "<tool_call>\n{\"name\": \"read_person\", \"arguments\": {\"face_id\": \"face_042\"}}\n</tool_call>"},
    {"role": "tool", "content": "<tool_response>\n{\"name\": \"read_person\", \"result\": {\"name\": \"Maria\", \"relationship\": \"granddaughter\", \"bio\": \"8 years old, loves drawing, visits every Sunday. You used to take her fishing at the lake.\"}}\n</tool_response>"},
    {"role": "assistant", "content": "That's Maria, your granddaughter! She's 8 years old and loves drawing. She visits you every Sunday. You two used to go fishing together at the lake."}
  ]
}
```

## SQLite Tools Implementation

### Database Initialization
- Create tables on first run if they don't exist.
- Use WAL mode for concurrent read/write.
- sqlite-vec extension for vector similarity search.

### Face Embedding Search
Face embeddings are **128-dimensional** float32 vectors from MobileFaceNet ONNX (1MB model).
NOT CLIP (512-d, 400MB) or ArcFace-R100 (512-d, 250MB) — those are too large for mobile.

```python
def search_face(db, embedding: list[float], threshold: float = 0.85) -> dict | None:
    """Search for matching face using MobileFaceNet 128-d embedding."""
    # vec_distance_cosine returns distance (0 = identical, 2 = opposite)
    # threshold 0.85 similarity = 0.15 distance
    # MobileFaceNet outputs normalized 128-d float32 vectors (~512 bytes)
    results = db.execute("""
        SELECT p.*, vec_distance_cosine(fe.embedding, ?) as distance
        FROM face_embeddings fe
        JOIN persons p ON fe.person_id = p.id
        WHERE distance < ?
        ORDER BY distance ASC
        LIMIT 1
    """, [embedding, 1.0 - threshold])
    return results.fetchone()
```

### Tool Execution Safety
- All SQL queries use parameterized statements (NO string formatting).
- Tool execution is sandboxed — only predefined tools can run.
- Read operations never modify data.
- Write operations log to encounters table for audit.

## Data Categories for Synthetic Generation

1. **Person recognition** (40% of data): "Who is this?", person arrives, face match
2. **Orientation** (20%): "Where am I?", "What time is it?", scene description
3. **Medication** (15%): Reminders, confirmation, missed dose alerts
4. **Memory sharing** (15%): Tell me about [person], shared stories, photo albums
5. **Caregiver alerts** (10%): Confusion detection, wandering, emergency
