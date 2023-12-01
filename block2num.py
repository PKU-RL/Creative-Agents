# Desc: Converts a block to the number

def block2num(block_name):
    """
    Converts a block to the number
    """
    if "grass" in block_name:
        return 64
    elif "dirt" in block_name:
        return 64
    elif "stone" in block_name:
        return 320
    elif "sand" in block_name:
        return 128
    elif "glass" in block_name:
        return 64
    elif "plank" in block_name:
        return 320
    elif "brick" in block_name:
        return 320
    elif "door" in block_name:
        return 64
    elif "torch" in block_name:
        return 64
    elif "ladder" in block_name:
        return 64
    elif "fence" in block_name:
        return 64
    elif "stairs" in block_name:
        return 128
    elif "slab" in block_name:
        return 256
    elif "bed" in block_name:
        return 1
    elif "rail" in block_name:
        return 64
    elif "lantern" in block_name:
        return 64
    elif "quartz" in block_name:
        return 320
    elif "red" in block_name:
        return 64
    elif "terracotta" in block_name:
        return 64
    elif "purpur" in block_name:
        return 128
    elif "wool" in block_name:
        return 256
    elif "concrete" in block_name:
        return 320
    elif "carpet" in block_name:
        return 64
    elif "clay" in block_name:
        return 64
    elif "leaves" in block_name:
        return 64
    elif "log" in block_name:
        return 320
    elif "stained" in block_name:
        return 64
    elif "glazed" in block_name:
        return 64
    elif "shulker" in block_name:
        return 64
    elif "end" in block_name:
        return 64
    elif "obsidian" in block_name:
        return 256
    elif "ice" in block_name:
        return 256
    elif "snow" in block_name:
        return 128
    elif "pumpkin" in block_name:
        return 64
    elif "melon" in block_name:
        return 64
    elif "nether" in block_name:
        return 256
    elif "glowstone" in block_name:
        return 64
    elif "sea" in block_name:
        return 64
    elif "soul" in block_name:
        return 64
    else:
        return 64