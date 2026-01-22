import json
import re

input_file = "confessions_for_training.jsonl"
output_file = "confessions_for_training_YUGOGPT_cleaned.jsonl"

with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    for line in f_in:
        data = json.loads(line)
        text = data["text"]
        
        # Remove the second bracketed group (after engagement_score)
        # This assumes the pattern: [something] [topic_to_keep] [topic_to_remove] ...
        bracketed_parts = re.findall(r'\[.*?\]', text)
        if len(bracketed_parts) > 2:
            # Remove the second extra topic (3rd bracketed part)
            text = text.replace(bracketed_parts[2], "").strip()
            # Remove extra spaces left behind
            text = re.sub(r'\s{2,}', ' ', text)
        
        data["text"] = text
        f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

print(f"Done! Cleaned JSONL saved to {output_file}")