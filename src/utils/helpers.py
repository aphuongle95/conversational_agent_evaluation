def replace_special_chars(text: str) -> str:
    """Replace German special characters with their ASCII equivalents."""
    if not text:
        return ""
        
    # First decode any potential encoding issues
    try:
        text = text.encode('latin1').decode('utf-8')
    except:
        pass
        
    replacements = {
        'ü': 'ue', 'Ü': 'Ue',
        'ä': 'ae', 'Ä': 'Ae',
        'ö': 'oe', 'Ö': 'Oe',
        'ß': 'ss', 'ß': 'ss'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text 