"""Local Chrome/CDP integration QA. QA-only dependency: websocket-client.

Start Chrome with remote debugging on 9225 and the viewer on 8765 first.
"""
import argparse
import base64
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

sys.path.insert(0, '/tmp/ms3d-viewer-qa')
import websocket

ROOT = Path(__file__).resolve().parents[1]
opener = build_opener(ProxyHandler({}))


def http(url, method='GET'):
    return opener.open(Request(url, method=method), timeout=20)


class CDP:
    def __init__(self, url):
        self.socket = websocket.create_connection(url, timeout=30, suppress_origin=True)
        self.sequence, self.events = 0, []

    def call(self, method, **params):
        self.sequence += 1
        self.socket.send(json.dumps(dict(id=self.sequence, method=method, params=params)))
        while True:
            response = json.loads(self.socket.recv())
            if response.get('id') == self.sequence:
                if 'error' in response:
                    raise RuntimeError(response['error'])
                return response.get('result', {})
            self.events.append(response)

    def evaluate(self, expression):
        result = self.call('Runtime.evaluate', expression=expression, returnByValue=True, awaitPromise=True)
        if 'exceptionDetails' in result:
            raise RuntimeError(result['exceptionDetails'])
        return result.get('result', {}).get('value')

    def wait(self, expression, timeout=25):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            value = self.evaluate(expression)
            if value:
                return value
            time.sleep(.15)
        raise AssertionError(f'Timeout waiting for {expression}')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    parser.add_argument('--debug-url', default='http://127.0.0.1:9225')
    parser.add_argument('--output', default='runs/local_viewer/qa')
    args = parser.parse_args()
    output = ROOT / args.output
    downloads = output / 'downloads' / time.strftime('%Y%m%d-%H%M%S')
    downloads.mkdir(parents=True, exist_ok=True)
    checks = []
    def passed(name):
        checks.append(name)
        print('PASS:', name, flush=True)

    with http(args.url+'/api/catalog') as response:
        catalog = json.load(response)
    assert len(catalog['frames']) == 204 and len(catalog['cases']) == 11
    with http(args.url+'/api/frame/100') as response:
        frame = json.load(response)
    with http(args.url+'/api/points/100?max_points=1000') as response:
        assert len(response.read()) == 12000 and response.headers['X-Coordinate-Frame'] == 'vehicle'
    for path, status in [('/api/frame/999', 400), ('/api/points/0?max_points=999999', 400),
                         ('/server.py', 404), ('/../../checkpoints/lidar-only-det.pth', 404), ('/api/missing', 404)]:
        try:
            http(args.url+path)
            raise AssertionError('Expected HTTP failure: '+path)
        except HTTPError as error:
            assert error.code == status
    passed('HTTP catalog/frame/binary points, invalid input and static file boundary')
    with http(args.debug_url+'/json/list') as response:
        previous_targets = json.load(response)
    for previous_target in previous_targets:
        if previous_target.get('type') == 'page' and previous_target.get('url', '').startswith(args.url):
            with http(args.debug_url+'/json/close/'+previous_target['id']) as response:
                response.read()
    with http(args.debug_url+'/json/new?about:blank', 'PUT') as response:
        target = json.load(response)
    cdp = CDP(target['webSocketDebuggerUrl'])
    for domain in ['Page', 'Runtime', 'Network', 'Log']:
        cdp.call(domain+'.enable')
    cdp.call('Network.setCacheDisabled', cacheDisabled=True)
    cdp.call('Emulation.setDeviceMetricsOverride', width=1560, height=1100, deviceScaleFactor=1, mobile=False)
    cdp.call('Browser.setDownloadBehavior', behavior='allow', downloadPath=str(downloads))
    cdp.call('Page.navigate', url=args.url+'/?frame=100&case=fusion3')
    snapshot = "window.__MS3D_VIEWER__?.snapshot()"
    cdp.wait(f'{snapshot}?.ready && !{snapshot}.loading')
    initial = cdp.evaluate(snapshot)
    expected = len([d for d in frame['cases']['fusion3'] if d['score'] >= .3])
    assert initial['webgl'] and initial['index'] == 100 and initial['visibleBoxes'] == expected
    assert initial['displayedPoints'] > 10000
    passed('Real Chrome module loading, WebGL scene, BEV and correct frame-100 box count')

    def event(element, value, name='input'):
        return cdp.evaluate(f"document.getElementById({json.dumps(element)}).value={json.dumps(value)};document.getElementById({json.dumps(element)}).dispatchEvent(new Event({json.dumps(name)},{{bubbles:true}}));true")
    def click(selector):
        return cdp.evaluate(f'document.querySelector({json.dumps(selector)}).click();true')
    event('score', '.8')
    assert cdp.evaluate(snapshot)['visibleBoxes'] == len([d for d in frame['cases']['fusion3'] if d['score'] >= .8])
    event('groupFilter', 'single')
    event('sourceFilter', 'bevfusion')
    expected_single = len([d for d in frame['cases']['fusion3'] if d['score'] >= .8 and d['kind']=='single' and 'bevfusion' in d['sources']])
    assert cdp.evaluate(snapshot)['visibleBoxes'] == expected_single
    click('#resetFilters')
    click('.class-check[value="1"]')
    assert cdp.evaluate(snapshot)['visibleBoxes'] == len([d for d in frame['cases']['fusion3'] if d['class_id'] != 1 and d['score'] >= .3])
    click('#resetFilters')
    passed('Score, class, singleton and contributing-source filters with exact API count checks')
    click('#compareEnabled')
    assert cdp.evaluate(snapshot)['comparisonBoxes'] == len([d for d in frame['cases']['fusion2'] if d['score'] >= .3])
    click('#compareEnabled')
    for case in ['bevfusion', 'kbf2', 'fusion2_relaxed', 'ms3d_final', 'centerpoint', 'fusion_bcv', 'ms3d_bcv', 'fusion3']:
        click(f'.case-option[data-case="{case}"]')
        assert cdp.evaluate(snapshot)['visibleBoxes'] == len([d for d in frame['cases'][case] if d['score'] >= .3])
    passed('Case switching and synchronized secondary-case overlay')
    click('#boxTable tr[data-id]')
    assert cdp.evaluate(snapshot)['selected']
    before_camera = cdp.evaluate(snapshot)['camera']
    click('#focusSelected')
    assert cdp.evaluate(snapshot)['camera'] != before_camera
    click('[data-camera="top"]')
    assert abs(cdp.evaluate(snapshot)['camera'][2]-85) < 1e-3
    click('[data-camera="iso"]')
    click('#clearSelection')
    candidate = next(d for d in frame['cases']['fusion3'] if d['score']>=.3 and abs(d['box'][0])<20 and abs(d['box'][1])<20)
    rect = cdp.evaluate("JSON.parse(JSON.stringify(document.getElementById('bevCanvas').getBoundingClientRect()))")
    scale = min(rect['width'], rect['height'])/80
    x = rect['x']+rect['width']/2-candidate['box'][1]*scale
    y = rect['y']+rect['height']/2-candidate['box'][0]*scale
    cdp.call('Input.dispatchMouseEvent', type='mousePressed', x=x, y=y, button='left', clickCount=1)
    cdp.call('Input.dispatchMouseEvent', type='mouseReleased', x=x, y=y, button='left', clickCount=1)
    assert cdp.evaluate(snapshot)['selected']
    passed('Box-table selection, BEV hit selection, selected-box inspector and camera focus/presets')
    rect = cdp.evaluate("JSON.parse(JSON.stringify(document.querySelector('#threeWrap canvas').getBoundingClientRect()))")
    before_camera = cdp.evaluate(snapshot)['camera']
    x, y = rect['x']+rect['width']/2, rect['y']+rect['height']/2
    cdp.call('Input.dispatchMouseEvent', type='mousePressed', x=x, y=y, button='left', clickCount=1)
    for step in range(1, 9):
        cdp.call('Input.dispatchMouseEvent', type='mouseMoved', x=x+step*8, y=y+step*4, button='left', buttons=1)
    cdp.call('Input.dispatchMouseEvent', type='mouseReleased', x=x+64, y=y+32, button='left', clickCount=1)
    time.sleep(.3)
    assert cdp.evaluate(snapshot)['camera'] != before_camera
    passed('Actual pointer-driven OrbitControls rotation')
    for index in [101, 102, 103]:
        event('frameInput', index, 'change')
    cdp.wait(f'{snapshot}.index===103 && !{snapshot}.loading')
    click('#previous')
    cdp.wait(f'{snapshot}.index===102 && !{snapshot}.loading')
    click('#next')
    cdp.wait(f'{snapshot}.index===103 && !{snapshot}.loading')
    passed('Rapid frame changes cancel stale requests; previous/next navigation')
    event('pointBudget', '20000', 'change')
    cdp.wait(f'!{snapshot}.loading && {snapshot}.displayedPoints<=20000')
    low_count = cdp.evaluate(snapshot)['displayedPoints']
    event('pointBudget', '120000', 'change')
    cdp.wait(f'!{snapshot}.loading && {snapshot}.displayedPoints>{low_count}')
    full_count = cdp.evaluate(snapshot)['displayedPoints']
    event('zMin', '1.0')
    assert cdp.evaluate(snapshot)['displayedPoints'] < full_count
    event('zMin', '-3')
    event('pointBudget', '60000', 'change')
    cdp.wait(f'!{snapshot}.loading')
    passed('Point budget and height clipping update both BEV and real 3D geometry')
    previous = cdp.evaluate(snapshot)['index']
    click('#play')
    cdp.wait(f'{snapshot}.index!=={previous} && !{snapshot}.loading')
    click('#play')
    passed('Playback advances frames and pauses')
    event('frameInput', 100, 'change')
    cdp.wait(f'{snapshot}.index===100 && !{snapshot}.loading')
    click('[data-camera="iso"]')
    click('#boxTable tr[data-id]')
    before_files = set(downloads.iterdir())
    click('#exportJson')
    deadline=time.monotonic()+10
    while time.monotonic()<deadline and not [p for p in downloads.glob('*.json') if p not in before_files]:
        time.sleep(.1)
    json_file = next(p for p in downloads.glob('*.json') if p not in before_files)
    exported = json.loads(json_file.read_text())
    assert exported['coordinate_frame']=='vehicle' and len(exported['detections'])==expected
    before_files = set(downloads.iterdir())
    click('#screenshot')
    deadline=time.monotonic()+15
    while time.monotonic()<deadline and not [p for p in downloads.glob('*.png') if p not in before_files]:
        time.sleep(.1)
    screenshot_file = next(p for p in downloads.glob('*.png') if p not in before_files)
    assert screenshot_file.stat().st_size>10000
    passed('UI JSON export and combined BEV/3D screenshot download')
    layout = cdp.call('Page.getLayoutMetrics')['cssContentSize']
    capture = cdp.call('Page.captureScreenshot', format='png', captureBeyondViewport=True,
                       clip=dict(x=0,y=0,width=layout['width'],height=layout['height'],scale=1))
    (output/'desktop.png').write_bytes(base64.b64decode(capture['data']))
    for width, height, mobile in [(1024,900,False),(390,844,True)]:
        cdp.call('Emulation.setDeviceMetricsOverride', width=width,height=height,deviceScaleFactor=1,mobile=mobile)
        time.sleep(.5)
        assert cdp.evaluate('document.documentElement.scrollWidth<=window.innerWidth+1')
        assert cdp.evaluate(snapshot)['webgl']
        capture = cdp.call('Page.captureScreenshot', format='png',captureBeyondViewport=False)
        (output/f'viewport_{width}.png').write_bytes(base64.b64decode(capture['data']))
    passed('1024px and 390px responsive layouts without horizontal page overflow')
    exceptions=[e for e in cdp.events if e.get('method')=='Runtime.exceptionThrown']
    console_errors=[e for e in cdp.events if e.get('method')=='Runtime.consoleAPICalled' and e['params']['type']=='error']
    external=[e['params']['request']['url'] for e in cdp.events if e.get('method')=='Network.requestWillBeSent'
              and urlsplit(e['params']['request']['url']).scheme in ['http','https']
              and not e['params']['request']['url'].startswith(args.url)]
    assert not exceptions and not console_errors, (exceptions,console_errors)
    assert not external, external
    passed('No browser runtime/console errors and no external asset/data requests')
    report=dict(status='PASS',checks=checks,browser='Installed Google Chrome, headless, real WebGL via SwiftShader',
                default_snapshot=initial,exported_json=str(json_file.relative_to(ROOT)),
                exported_screenshot=str(screenshot_file.relative_to(ROOT)),
                runtime_exceptions=len(exceptions),console_errors=len(console_errors),external_requests=external,
                screenshots=[str((output / p).relative_to(ROOT)) for p in ['desktop.png','viewport_1024.png','viewport_390.png']])
    (output/'browser_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    cdp.socket.close()


if __name__=='__main__':
    main()
