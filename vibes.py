import os
from functools import reduce


def load_text_files(directory):
    text_files = []

    # Check if directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' not found")

    # List only files in the immediate directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        # Check if it's a file (not a directory) and ends with .txt
        if os.path.isfile(file_path) and filename.endswith('.txt'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    text_files.append({
                        'basename': os.path.splitext(filename)[0],
                        'content': content
                    })
            except Exception as e:
                print(f"Error reading file {file_path}: {str(e)}")

    return text_files


def parse_line(line):
    # Extract the timestamp part and text part
    timestamp_part, text = line.split(']', 1)
    # Remove the brackets and split the timestamps
    timestamps = timestamp_part.strip('[]').split(' --> ')

    # Convert timestamps to seconds (assumes format HH:MM:SS.mmm)
    def timestamp_to_seconds(ts):
        h, m, s = ts.split(':')
        return float(h) * 3600 + float(m) * 60 + float(s)

    start_index = timestamp_to_seconds(timestamps[0])
    end_index = timestamp_to_seconds(timestamps[1])
    text = text.strip()

    return {
        'start_index': start_index,
        'end_index': end_index,
        'text': text
    }


def combine_slices(slices, min_length=100):
    def combine_reducer(acc, slice):
        if not acc:
            return [(slice,)]
        last_group = acc[-1]
        combined_text = ' '.join(s["text"] for s in last_group)
        if len(combined_text) >= min_length:
            return acc + [(slice,)]
        return acc[:-1] + [(*last_group, slice)]

    return [
        {
            "start_index": group[0]["start_index"],
            "end_index": group[-1]["end_index"],
            "text": ' '.join(s["text"] for s in group)
        }
        for group in reduce(combine_reducer, slices, [])
    ]
