import ssl, urllib.request
urls = [
    "https://raw.githubusercontent.com/sumdeusvitae/FaceSwap_v01/main/inswapper_128.onnx",
    "https://github.com/omarsar/inswapper/raw/main/inswapper_128.onnx",
    "https://huggingface.co/cloneofsimo/inswapper_v3/resolve/main/inswapper_128.onnx",
    "https://huggingface.co/spaces/karfly/inswapper/resolve/main/inswapper_128.onnx"
]
ctx = ssl.create_default_context()
for url in urls:
    print('\nURL:', url)
    try:
        req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ctx, timeout=30) as r:
            data = r.read(1024)
            size = r.getheader('Content-Length')
            print('status:', getattr(r, 'status', 'N/A'))
            print('content-length header:', size)
            try:
                txt = data.decode('utf-8', errors='ignore')
                snippet = txt[:200]
                print('snippet (utf8):', snippet.replace('\n', ' '))
            except Exception as e:
                print('snippet decode error', e)
    except Exception as e:
        print('ERR', type(e).__name__, e)
