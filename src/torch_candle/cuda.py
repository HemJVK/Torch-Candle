import candle

def is_available():
    # Check if candle was built with cuda
    try:
        candle.Tensor([0]).to_device('cuda')
        return True
    except:
        return False

def device_count():
    # Placeholder
    return 1 if is_available() else 0

def get_device_name(device=None):
    if is_available():
        return "NVIDIA GPU (via Candle)"
    return "CPU"
