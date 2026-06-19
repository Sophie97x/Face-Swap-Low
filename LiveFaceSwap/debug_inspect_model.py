import pathlib
p = pathlib.Path(__file__).resolve().parent / 'models' / 'inswapper_128.onnx'
print('path:', p)
print('exists:', p.exists())
if p.exists():
    s = p.stat().st_size
    print('size:', s)
    data = p.read_bytes()
    head = data[:512]
    try:
        text = head.decode('utf-8', errors='ignore')
        print('head (utf-8 snippet):')
        print(text[:512])
    except Exception as e:
        print('could not decode head:', e)
    print('head hex:', head[:32].hex())
    # quick check if file looks like HTML
    if b'<html' in head.lower() or b'<!doctype html' in head.lower():
        print('File looks like HTML (probably download error page).')
    else:
        print('File does not look like HTML at start.')
else:
    print('file missing')
