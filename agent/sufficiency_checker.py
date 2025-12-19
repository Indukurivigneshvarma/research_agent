def is_sufficient(query: str, knowledge_chunks: list) -> bool:
    if len(knowledge_chunks) < 3:
        return False

    unique_points = set(knowledge_chunks)
    if len(unique_points) < len(knowledge_chunks) * 0.6:
        return True

    return False
