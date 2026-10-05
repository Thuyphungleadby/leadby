import cv2, numpy as np, onnxruntime as ort, subprocess, sys, json, time
inp, out, box, t0, t1 = sys.argv[1], sys.argv[2], json.loads(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
only = [int(x) for x in sys.argv[6].split(',')] if len(sys.argv) > 6 else None
so = ort.SessionOptions(); so.intra_op_num_threads = 4
sess = ort.InferenceSession('lama/lama_fp32.onnx', so)
cap = cv2.VideoCapture(inp); fps = cap.get(cv2.CAP_PROP_FPS); frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(f)
n = len(frames); H, W = frames[0].shape[:2]
bx, by, bw, bh = box; X0, Y0 = int(bx * W), int(by * H); X1, Y1 = int((bx + bw) * W), int((by + bh) * H)
K = np.ones((3, 3), np.uint8)
def text_mask(k):
    f = frames[k][Y0:Y1, X0:X1].astype(np.int16); st = np.zeros(f.shape[:2], bool)
    for d in (-15, -8, 8, 15):
        j = k + d
        if 0 <= j < n:
            g = frames[j][Y0:Y1, X0:X1].astype(np.int16)
            st |= np.abs(f - g).max(2) < 14
    m = st.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, K)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    full = np.zeros((H, W), np.uint8); full[Y0:Y1, X0:X1] = m
    return full
# context crop: square of side W centred on the box
side = min(W, H); cy = (Y0 + Y1) // 2; CY0 = int(np.clip(cy - side // 2, 0, H - side))
def inpaint(f, m):
    c = f[CY0:CY0 + side, :side]; mc = m[CY0:CY0 + side, :side]
    img = cv2.resize(c, (512, 512), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) / 255
    mm = (cv2.resize(mc, (512, 512), interpolation=cv2.INTER_NEAREST) > 127).astype(np.float32)
    o = sess.run(None, {'image': img.transpose(2, 0, 1)[None], 'mask': mm[None, None]})[0][0]
    o = o.transpose(1, 2, 0)
    if o.max() <= 1.5: o = o * 255
    o = cv2.resize(np.clip(o, 0, 255).astype(np.uint8)[:, :, ::-1], (side, side), interpolation=cv2.INTER_CUBIC)
    a = cv2.GaussianBlur(mc, (7, 7), 0).astype(np.float32)[..., None] / 255
    r = f.copy(); r[CY0:CY0 + side, :side] = (o * a + c * (1 - a)).astype(np.uint8)
    return r
if only:
    for k in only:
        m = text_mask(k); t = time.time(); r = inpaint(frames[k], m); print('frame', k, round(time.time() - t, 2), 's')
        vis = frames[k].copy(); vis[m > 0] = (vis[m > 0] * 0.4 + np.array([0, 0, 255]) * 0.6).astype(np.uint8)
        cv2.imwrite(f'{out}_{k}.png', np.hstack([frames[k], vis, r]))
    sys.exit()
ff = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(fps), '-i', '-',
    '-i', inp, '-map', '0:v', '-map', '1:a?', '-c:v', 'libx264', '-crf', '17', '-preset', 'medium', '-pix_fmt', 'yuv420p', '-c:a', 'copy', '-movflags', '+faststart', '-shortest', out], stdin=subprocess.PIPE)
t = time.time()
for k in range(n):
    f = frames[k]
    if t0 <= k / fps <= t1:
        m = text_mask(k)
        if m.any(): f = inpaint(f, m)
    ff.stdin.write(f.tobytes())
    if k % 60 == 0: print(f'{k}/{n} {time.time() - t:.0f}s', flush=True)
ff.stdin.close(); ff.wait(); print('done', out)
