"""Check the lesson notebook's real data-to-prompt path without cloud clients."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace


# Intended destination in the repository: tests/test_rag_notebook.py.
NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1]
    / "15-rag-and-vector-databases"
    / "notebook-rag-vector-databases.ipynb"
)


def test_local_rag_documents_reach_the_response_prompt():
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    data_paths = None
    chatbot_function = None
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        # Parse only the two relevant cells. Never execute setup, install,
        # credentials, embedding requests, vectorization, or notebook outputs.
        if "data_paths =" in source:
            for node in ast.parse(source).body:
                if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "data_paths"
                    for target in node.targets
                ):
                    data_paths = ast.literal_eval(node.value)
        if "def chatbot(" in source:
            chatbot_function = next(
                node
                for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == "chatbot"
            )

    assert data_paths is not None and chatbot_function is not None
    documents = [
        (NOTEBOOK_PATH.parent / relative_path).read_text(encoding="utf-8")
        for relative_path in data_paths
    ]
    assert len(documents) == 3 and all(documents)

    question = "What does the retrieved course material explain?"
    query_vector = [0.125, 0.25]
    retrieval_order = [2, 0, 1]
    requests = []

    def create_embeddings(text):
        assert text == question
        return query_vector

    def kneighbors(queries):
        assert queries == [query_vector]
        return [[0.1, 0.2, 0.3]], [retrieval_order]

    def create_response(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(output_text="OFFLINE_RESPONSE_SENTINEL")

    namespace = {
        "create_embeddings": create_embeddings,
        "nbrs": SimpleNamespace(kneighbors=kneighbors),
        "flattened_df": {"chunks": SimpleNamespace(iloc=documents)},
        "client": SimpleNamespace(responses=SimpleNamespace(create=create_response)),
        "chat_deployment": "offline-test-deployment",
    }
    module = ast.Module(body=[chatbot_function], type_ignores=[])
    exec(compile(module, str(NOTEBOOK_PATH), "exec"), namespace)

    assert namespace["chatbot"](question) == "OFFLINE_RESPONSE_SENTINEL"
    assert len(requests) == 1
    user_message = requests[0]["input"][-1]
    assert user_message["role"] == "user"
    content = user_message["content"]
    for document in documents:
        assert document in content
    positions = [content.index(documents[index]) for index in retrieval_order]
    assert positions == sorted(positions)
    assert content.endswith(question)
