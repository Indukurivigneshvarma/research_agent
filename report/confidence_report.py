def build_confidence_report(knowledge):
    if len(knowledge) > 5:
        return "High confidence based on multiple independent sources."
    return "Moderate confidence due to limited sources."
