# funasr/utils/tables.py

frontend_classes = {}

def register(name, value=None):
    def _register(cls):
        if name in globals():
            getattr(globals()[name], 'update')({value: cls})
        else:
            globals()[name] = {value: cls}
        return cls
    return _register