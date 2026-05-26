class Net:
    """caffe2 mobile execution Net target stub."""
    def __init__(self, name="default"):
        self.name = name
        self.ops = []
        
    def add_op(self, op_type, inputs, outputs):
        self.ops.append((op_type, inputs, outputs))

def is_mobile_available():
    """Return whether edge compiler runtime features are packaged."""
    return False

def transform_to_mobile(script_module):
    """Placeholder transforming a ScriptModule to edge-friendly runtime representations."""
    return script_module
