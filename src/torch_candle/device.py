import candle

class device:
    def __init__(self, type, index=None):
        if isinstance(type, str):
            if ':' in type:
                type, index = type.split(':')
                index = int(index)
            self.type = type
            self.index = index
        else:
            self.type = type.type if hasattr(type, 'type') else str(type)
            self.index = type.index if hasattr(type, 'index') else None

        # Map to candle device string
        if self.type == 'cpu':
            self._candle_device = 'cpu'
        elif self.type == 'cuda':
            self._candle_device = f'cuda:{self.index}' if self.index is not None else 'cuda'
        elif self.type == 'mps':
            self._candle_device = 'metal'
        else:
            self._candle_device = self.type

    def __repr__(self):
        if self.index is not None:
            return f"device(type='{self.type}', index={self.index})"
        return f"device(type='{self.type}')"

    def __str__(self):
        if self.index is not None:
            return f"{self.type}:{self.index}"
        return self.type
