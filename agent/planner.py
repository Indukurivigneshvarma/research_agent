def plan_search(query: str, iteration: int) -> str:
    if iteration == 1:
        return query
    elif iteration == 2:
        return f"{query} statistics reports"
    else:
        return f"{query} expert opinions limitations"
