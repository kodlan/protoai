import re
import difflib
import openai
import numpy as np
import faiss

# Set your OpenAI API key
openai.api_key = "place your key here"

def extract_messages(content: str) -> dict:
    """
    Extracts proto messages from file content.
    Returns a dictionary mapping message names to the full message block.
    """
    # Regex assumes messages start with "message MessageName {" and end with a closing brace on its own line.
    pattern = r"(message\s+(\w+)\s*\{.*?\n\})"
    matches = re.findall(pattern, content, flags=re.DOTALL)
    messages = {match[1]: match[0] for match in matches}
    return messages

def get_openai_embedding(text: str, model: str = "text-embedding-ada-002") -> np.ndarray:
    """
    Uses OpenAI's embedding API (text-embedding-ada-002) to generate an embedding for the given text.
    """
    response = openai.embeddings.create(input=[text], model=model)
    embedding = response.data[0].embedding
    return np.array(embedding, dtype="float32")

def compute_message_diffs(file1: str, file2: str) -> (dict, list, list):
    """
    Reads two proto files, extracts messages, and computes a diff per message.
    Returns:
      - diff_mapping: a dict mapping message name to its diff text.
      - embeddings_list: a list of embeddings (one per non-empty message diff).
      - index_to_key: a list mapping FAISS index positions to message names.
    """
    # Read file contents
    with open(file1, 'r') as f:
        content1 = f.read()
    with open(file2, 'r') as f:
        content2 = f.read()

    # Extract messages from both files
    messages1 = extract_messages(content1)
    messages2 = extract_messages(content2)

    # Combine keys to process all messages
    all_keys = set(messages1.keys()).union(set(messages2.keys()))

    diff_mapping = {}   # key: message name, value: diff text
    embeddings_list = []  # list of embeddings for each non-empty diff
    index_to_key = []   # mapping from embedding index to message name

    for key in all_keys:
        if key in messages1 and key in messages2:
            # Compute unified diff for messages that exist in both versions
            diff_lines = list(difflib.unified_diff(
                messages1[key].splitlines(),
                messages2[key].splitlines(),
                fromfile=f"{file1}:{key}",
                tofile=f"{file2}:{key}",
                lineterm=''
            ))
            diff_text = "\n".join(diff_lines)
        elif key in messages2 and key not in messages1:
            # New message added
            diff_text = f"New message '{key}' added:\n" + messages2[key]
        elif key in messages1 and key not in messages2:
            # Message removed
            diff_text = f"Message '{key}' removed:\n" + messages1[key]
        else:
            diff_text = ""

        # Store the diff text (for later lookup/display)
        diff_mapping[key] = diff_text

        # Skip empty diffs
        if not diff_text.strip():
            continue

        # Get embedding for the non-empty diff text
        emb = get_openai_embedding(diff_text)
        embeddings_list.append(emb)
        index_to_key.append(key)

    return diff_mapping, embeddings_list, index_to_key

def main():
    # Paths to two versions of your proto file
    file_v1 = '/proto/pv1.proto'
    file_v2 = '/proto/pv2.proto'

    # Compute diffs and embeddings for each proto message
    diff_mapping, embeddings_list, index_to_key = compute_message_diffs(file_v1, file_v2)

    if not embeddings_list:
        print("No differences found or all diffs are empty.")
        return

    # Stack embeddings into a numpy array for FAISS (each row is one embedding)
    embeddings_np = np.vstack(embeddings_list)

    # Create a FAISS index (using L2 distance)
    embedding_dim = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_np)
    print("Diff embeddings for each message have been indexed in FAISS.")





    # Example query: Adjust the query as needed
    query_text = "What new messages were added?"
    query_embedding = get_openai_embedding(query_text)
    query_embedding = np.expand_dims(query_embedding, axis=0)

    # Search the FAISS index for the top 3 matches
    distances, indices = index.search(query_embedding, k=3)
    print("Query:", query_text)
    print("\nTop matching message diffs:")
    for i in range(len(indices[0])):
        msg_key = index_to_key[indices[0][i]]
        print(f"\n--- Message: {msg_key} (Distance: {distances[0][i]:.4f}) ---")
        print(diff_mapping[msg_key])




    # Example query: Adjust the query as needed
    query_text = "What fields were renamed?"
    query_embedding = get_openai_embedding(query_text)
    query_embedding = np.expand_dims(query_embedding, axis=0)

    # Search the FAISS index for the top 3 matches
    distances, indices = index.search(query_embedding, k=3)
    print("Query:", query_text)
    print("\nTop matching diffs:")
    for i in range(len(indices[0])):
        msg_key = index_to_key[indices[0][i]]
        print(f"\n--- Message: {msg_key} (Distance: {distances[0][i]:.4f}) ---")
        print(diff_mapping[msg_key])

if __name__ == "__main__":
    main()
