import json
import os
import sys

from google import genai
from google.genai import types


def generate(image_file):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.5-flash-preview-04-17"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    mime_type="image/jpeg",
                    data=open(image_file, 'rb').read(),
                ),
                types.Part.from_text(text="""explain in five sentences or less the meaning of this comic panel. don't mention that it's a cartoon, just explain it. your explanation should focus on the themes, irony, absurdism, and/or surrealism present. only use very specific statements in your explanation, not generalizations. also mention any novel and notable situations, objects, people, and environments but only if they are distinguishing. extract any text that's present, clearly delineating where each fragment comes from. there may be a caption under the image that should be extracted separately and clearly delineated in your output."""),
            ],
        )
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0.8,
        thinking_config = types.ThinkingConfig(
            thinking_budget=5000,
        ),
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type = genai.types.Type.OBJECT,
            properties = {
                "description": genai.types.Schema(
                    type = genai.types.Type.STRING,
                ),
                "caption": genai.types.Schema(
                    type = genai.types.Type.STRING,
                ),
                "other_text": genai.types.Schema(
                    type = genai.types.Type.ARRAY,
                    items = genai.types.Schema(
                        type = genai.types.Type.OBJECT,
                        required = ["location", "text"],
                        properties = {
                            "location": genai.types.Schema(
                                type = genai.types.Type.STRING,
                            ),
                            "text": genai.types.Schema(
                                type = genai.types.Type.STRING,
                            ),
                        },
                    ),
                ),
            },
        ),
    )

    return [
        chunk
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
    ]


if __name__ == '__main__':
    gen = generate(sys.argv[1])

    text = ""
    for g in gen:
        if g.text:
            text += g.text

    parsed = json.loads(text)
    usage = gen[-1].usage_metadata

    parsed["in"] = usage.prompt_token_count
    parsed["out"] = usage.candidates_token_count
    parsed["think"] = usage.thoughts_token_count

    output_file = os.path.splitext(sys.argv[1])[0] + ".json"
    with open(output_file, "w") as f:
        json.dump(parsed, f, indent=2)

