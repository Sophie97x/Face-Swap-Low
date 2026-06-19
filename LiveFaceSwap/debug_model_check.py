import pathlib, sys
try:
    import onnxruntime as ort
except Exception as e:
    print('onnxruntime import error:', e)
    sys.exit(0)

p = pathlib.Path(__file__).resolve().parent / 'models' / 'inswapper_128.onnx'
print('model path:', p)
print('exists:', p.exists())
if p.exists():
    try:
        s = ort.InferenceSession(str(p))
        inputs = [(i.name, i.shape, i.type) for i in s.get_inputs()]
        outputs = [(o.name, o.shape, o.type) for o in s.get_outputs()]
        print('inputs:', inputs)
        print('outputs:', outputs)
    except Exception as e:
        print('session load error:', type(e).__name__, e)
else:
    print('Model file not found.\nPlease download inswapper_128.onnx into the models/ directory.')
