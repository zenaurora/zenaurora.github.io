"""Animate the supplied Kodee artwork with a small, deterministic puppet rig."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SIZE, FRAMES, DURATION = 512, 48, 50
TAU = 2 * math.pi
PURPLE = (139, 87, 255, 255)
PINK = (255, 39, 103, 255)

# Original-image coordinates: head bounds, left/right hands, left/right feet.
POSES = [
    ('walking', '走路', (160, 115, 285, 226), [(130, 160), (300, 207), (194, 278), (268, 285)]),
    ('happy', '开心', (145, 115, 265, 217), [(107, 204), (316, 200), (158, 280), (274, 282)]),
    ('greeting', '挥手', (143, 108, 270, 218), [(129, 243), (294, 139), (174, 287), (235, 287)]),
    ('cute', '卖萌', (149, 113, 269, 221), [(82, 214), (285, 216), (145, 284), (233, 285)]),
    ('relaxing', '喝饮料', (150, 103, 285, 221), [(142, 252), (244, 256), (181, 294), (252, 294)]),
    ('running', '奔跑', (171, 121, 299, 240), [(101, 231), (335, 146), (134, 283), (354, 264)]),
    ('running-fast', '冲刺', (157, 125, 290, 239), [(115, 177), (328, 213), (105, 230), (291, 291)]),
    ('jumping', '跳起来', (151, 110, 274, 216), [(90, 132), (320, 127), (160, 264), (284, 250)]),
    ('jumping-playful', '蹦蹦跳跳', (160, 105, 278, 217), [(103, 135), (325, 132), (200, 291), (289, 254)]),
    ('sitting', '坐着晃腿', (144, 113, 265, 223), [(82, 213), (332, 214), (151, 257), (217, 283)]),
    ('joyful', '收到礼物', (166, 130, 287, 232), [(169, 222), (289, 222), (181, 282), (254, 282)]),
    ('sad', '垂头丧气', (148, 108, 271, 228), [(186, 269), (279, 269), (207, 282), (232, 289)]),
    ('crying', '揉眼睛', (164, 137, 284, 235), [(171, 180), (260, 184), (183, 305), (221, 305)]),
    ('holding', '拿着笔', (190, 101, 310, 218), [(182, 214), (250, 224), (244, 305), (280, 305)]),
    ('drawing', '画画中', (141, 111, 270, 227), [(129, 280), (288, 118), (189, 294), (230, 302)]),
    ('in-love', '心动', (158, 124, 288, 233), [(153, 130), (289, 252), (191, 279), (207, 302)]),
    ('in-love-pink', '爱你呀', (160, 123, 284, 227), [(119, 138), (322, 149), (186, 299), (253, 299)]),
    ('angry', '生气', (146, 130, 270, 231), [(140, 203), (280, 210), (175, 300), (245, 300)]),
    ('shocked', '震惊', (161, 131, 282, 231), [(176, 215), (262, 216), (196, 298), (255, 298)]),
    ('dancing', '跳舞', (162, 110, 284, 223), [(103, 200), (322, 203), (186, 300), (221, 298)]),
    ('ice-skates', '滑冰', (150, 115, 290, 235), [(125, 248), (341, 242), (191, 306), (279, 278)]),
    ('heart', '送你一颗心', (136, 108, 274, 225), [(113, 155), (268, 250), (119, 248), (203, 298)]),
]


def extract_art(path: Path, head) -> Image.Image:
    rgb = Image.open(path).convert('RGB')
    a = np.asarray(rgb).copy()
    # Thin original guide lines are not part of the character.
    for y in range(225, 360):
        row = a[y]
        purple = (row[:, 2] > row[:, 0]) & (row[:, 2] > row[:, 1] * 1.3)
        dim = purple & (row.max(axis=1) < 170) & (row.max(axis=1) > 22)
        if dim.sum() > 80:
            a[y, dim] = 0
    black = Image.fromarray(np.where(a.max(axis=2) < 18, 255, 0).astype('uint8')).copy()
    ImageDraw.floodfill(black, (0, 0), 0, border=0)
    # Preserve the black screen, including pupils, while opening arm/leg loops.
    remaining = np.asarray(black).copy() > 0
    components = []
    for y, x in zip(*np.where(remaining)):
        if not remaining[y, x]:
            continue
        stack, pixels = [(int(x), int(y))], []
        remaining[y, x] = False
        while stack:
            px, py = stack.pop()
            pixels.append((px, py))
            for nx, ny in [(px-1, py), (px+1, py), (px, py-1), (px, py+1)]:
                if 0 <= nx < 400 and 0 <= ny < 400 and remaining[ny, nx]:
                    remaining[ny, nx] = False
                    stack.append((nx, ny))
        cx = sum(p[0] for p in pixels) / len(pixels)
        cy = sum(p[1] for p in pixels) / len(pixels)
        if len(pixels) > 1000 and head[0] < cx < head[2] and head[1] < cy < head[3]:
            components.append(pixels)
    screen = Image.new('L', rgb.size)
    if components:
        pixels = sorted(max(components, key=len))
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        lower, upper = [], []
        for p in pixels:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        for p in reversed(pixels):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        ImageDraw.Draw(screen).polygon(lower[:-1] + upper[:-1], fill=255)
    alpha = np.where((a.max(axis=2) >= 18) | (np.asarray(screen) > 0), 255, 0).astype('uint8')
    # Soften only the outside silhouette.
    inside = np.asarray(Image.fromarray(alpha).filter(ImageFilter.MinFilter(3)))
    boundary = (alpha > 0) & (inside == 0)
    coverage = np.clip(a.max(axis=2) / 245, 0, 1)
    alpha[boundary] = np.maximum(1, (coverage[boundary] * 255)).astype('uint8')
    safe = np.maximum(coverage, 1 / 255)
    a[boundary] = np.minimum(255, a[boundary] / safe[boundary, None]).astype('uint8')
    rgba = np.dstack((a, alpha))
    rgba[alpha == 0, :3] = 0
    return Image.fromarray(rgba)


def find_eyes(art, head):
    arr = np.asarray(art)
    x0, y0, x1, y1 = head
    white = (arr[:, :, :3].min(axis=2) > 210) & (arr[:, :, 3] > 200)
    white[:y0 + 34] = False
    white[y1 - 9:] = False
    white[:, :x0 + 8] = False
    white[:, x1 - 8:] = False
    seen = np.zeros_like(white)
    boxes = []
    for y, x in zip(*np.where(white)):
        if seen[y, x]:
            continue
        stack, xs, ys = [(x, y)], [], []
        seen[y, x] = True
        while stack:
            px, py = stack.pop()
            xs.append(px); ys.append(py)
            for nx, ny in [(px-1, py), (px+1, py), (px, py-1), (px, py+1)]:
                if 0 <= nx < 400 and 0 <= ny < 400 and white[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((nx, ny))
        box = (min(xs)-2, min(ys)-2, max(xs)+3, max(ys)+3)
        if 14 < box[2]-box[0] < 48 and 14 < box[3]-box[1] < 48 and len(xs) > 55:
            boxes.append(box)
    return sorted(boxes) if len(boxes) == 2 else []


def blink(art, eyes, phase, enabled):
    if not enabled or not eyes:
        return art
    distance = min(abs(phase - .76), abs(phase - .76 + 1), abs(phase - .76 - 1))
    if distance > .045:
        return art
    openness = max(.09, distance / .045)
    out = art.copy()
    draw = ImageDraw.Draw(out)
    for box in eyes:
        x0, y0, x1, y1 = box
        eye = art.crop(box)
        draw.rectangle(box, fill=(0, 0, 0, 255))
        height = max(3, round(eye.height * openness))
        eye = eye.resize((eye.width, height), Image.Resampling.LANCZOS)
        out.alpha_composite(eye, (x0, round((y0 + y1 - height) / 2)))
    return out


def motion(name, t):
    s, c = math.sin(t), math.cos(t)
    s2 = math.sin(2*t)
    controls = [(0., 0.) for _ in range(4)]
    dx = dy = angle = 0.
    if name == 'walking':
        controls = [(6*s, -4*s), (-7*s, 5*s), (24*s, -10*c), (-24*s, 10*c)]
        dy, angle = -3*math.cos(2*t), 2*s
    elif name == 'happy':
        controls = [(-10*s, -17*s), (10*s, -17*s), (-9*s, 5*s), (9*s, -5*s)]
        dy, angle = -8*(1-c), 3*s
    elif name == 'greeting':
        controls = [(3*s, 2*s), (12*math.sin(3*t), -5*(1-math.cos(3*t))), (0, 0), (0, 0)]
        angle = 2*s
    elif name == 'cute':
        controls = [(10*s, 6*s2), (-8*s, -8*s2), (12*s, -6*c), (-7*s, 3*c)]
        angle, dy = 5*s, -3*c
    elif name == 'relaxing':
        controls = [(1.5*s, -4*(1-c)), (0, 2*s), (0, 0), (0, 0)]
        angle, dy = 1.5*s, 2*s
    elif name in ('running', 'running-fast'):
        controls = [(15*s2, -14*s2), (-15*s2, 14*s2), (27*s2, -14*math.cos(2*t)), (-27*s2, 14*math.cos(2*t))]
        dx, dy, angle = 5*s, -5*math.cos(4*t), 2.5*s2
    elif name in ('jumping', 'jumping-playful'):
        jump = (1-c)/2
        controls = [(-6*jump, -12*jump), (6*jump, -12*jump), (-12*jump, 13*jump), (12*jump, 13*jump)]
        dy, angle = -38*jump, (2 if name == 'jumping' else 6)*s
    elif name == 'sitting':
        controls = [(0, -2*s), (0, 2*s), (17*s, 5*c), (-18*s, -5*c)]
        angle = 1.5*s
    elif name == 'joyful':
        controls = [(0, -4*s), (0, -4*s), (-5*s, 4*s), (5*s, -4*s)]
        dy, angle = -5*(1-c), 3*s
    elif name == 'sad':
        controls = [(3*s, 4*c), (3*s, 4*c), (0, 1*s), (0, 1*s)]
        dy, angle = 3*s, 2*s
    elif name == 'crying':
        controls = [(0, 4*s2), (0, -4*s2), (0, 0), (0, 0)]
        dx, dy, angle = 2*s2, 3*c, 1.5*s2
    elif name == 'holding':
        controls = [(2*s, -4*s), (2*s, -4*s), (0, 0), (0, 0)]
        dy, angle = 2*s, 2*s
    elif name == 'drawing':
        controls = [(3*s, 2*c), (13*math.sin(2*t), 8*math.cos(3*t)), (4*s, 2*c), (-3*s, 2*c)]
        dy, angle = 2*s, 2*s
    elif name in ('in-love', 'in-love-pink'):
        controls = [(-8*s, -7*s2), (8*s, -7*s2), (7*s, -3*c), (-7*s, 3*c)]
        dy, angle = -6*(1-c), 5*s
    elif name == 'angry':
        controls = [(-4*math.sin(5*t), -2*c), (4*math.sin(5*t), -2*c), (-3*s2, 0), (3*s2, 0)]
        dx, dy, angle = 2.8*math.sin(5*t), -2*math.cos(4*t), 1.5*s2
    elif name == 'shocked':
        controls = [(-3*s, -5*s), (3*s, -5*s), (-6*s, 4*c), (6*s, -4*c)]
        dy, angle = -6*(1-c), 1.5*s2
    elif name == 'dancing':
        controls = [(-13*s, -18*c), (13*s, 18*c), (18*s, -8*c), (-18*s, 8*c)]
        dx, dy, angle = 8*s, -5*math.cos(2*t), 7*s
    elif name == 'ice-skates':
        controls = [(-3*s, 3*s), (6*s, -4*s), (-12*s, 2*c), (15*s, -10*c)]
        dx, dy, angle = 18*s, -2*math.cos(2*t), 4*s
    elif name == 'heart':
        controls = [(15*math.sin(3*t), -4*(1-math.cos(3*t))), (3*s, -5*(1-c)), (7*s, -3*c), (0, 0)]
        dy, angle = -4*(1-c), 3*s
    return controls, dx, dy, angle


def heart(draw, center, size, fill):
    cx, cy = center
    pts = []
    for t in np.linspace(0, TAU, 60):
        x = 16 * math.sin(t) ** 3
        y = 13*math.cos(t)-5*math.cos(2*t)-2*math.cos(3*t)-math.cos(4*t)
        pts.append((cx + x*size/32, cy - y*size/32))
    draw.polygon(pts, fill=fill)


def overlay(frame, name, phase):
    fx = Image.new('RGBA', frame.size)
    draw = ImageDraw.Draw(fx)
    if name in ('in-love', 'in-love-pink', 'heart'):
        for j in range(3):
            age = (phase + j/3) % 1
            fade = math.sin(math.pi*age)
            x = 103 + j*148 + 8*math.sin(age*TAU+j)
            y = 240 - age*150
            heart(draw, (x, y), 16+12*fade, (*PINK[:3], int(235*fade)))
    if name in ('sad', 'crying'):
        for j in range(2):
            age = (phase * 2 + j*.48) % 1
            x = 220 + j*60
            y = 245 + 90*age
            alpha = int(210*math.sin(math.pi*age))
            draw.ellipse((x-4, y-8, x+4, y+6), fill=(144, 104, 255, alpha))
    if name in ('running', 'running-fast', 'ice-skates'):
        for j in range(3):
            age = (phase*2 + j/3) % 1
            x, y = 132-55*age, 315+j*23
            draw.line((x, y, x+24+18*age, y), fill=(156, 110, 255, int(170*math.sin(math.pi*age))), width=3)
    if name in ('happy', 'joyful', 'jumping', 'jumping-playful', 'shocked'):
        for j, (x, y) in enumerate([(113, 145), (397, 168)]):
            alpha = int(220*max(0, math.sin(phase*TAU+j*.9)))
            r = 6+4*math.sin(phase*TAU+j)
            draw.line((x-r, y, x+r, y), fill=(179, 143, 255, alpha), width=3)
            draw.line((x, y-r, x, y+r), fill=(179, 143, 255, alpha), width=3)
    if name == 'drawing':
        # A looping stroke follows the writing hand and fades before resetting.
        alpha = int(190*math.sin(math.pi*phase))
        xs = np.linspace(331, 405, max(2, int(50*phase)))
        pts = [(float(x), 107+8*math.sin((x-331)/12)) for x in xs]
        draw.line(pts, fill=(174, 130, 255, alpha), width=3)
    return Image.alpha_composite(frame, fx)


def render_pose(index, pose):
    name, label, head, points = pose
    art = extract_art(ROOT/'source'/f'{index+1:02d}-{name}.png', head)
    eyes = find_eyes(art, head)
    alpha = art.getchannel('A')
    bbox = alpha.getbbox()
    if pose[0] == 'walking':
        print(f'Extracted silhouette: {bbox}', flush=True)
    center = ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)
    scale = min(1.65, 345/max(bbox[2]-bbox[0], bbox[3]-bbox[1]))
    # A 16px mesh gives independent local motion without redrawing the face.
    nodes = np.linspace(0, SIZE, 33)
    xx, yy = np.meshgrid(nodes, nodes)
    frames = []
    for f in range(FRAMES):
        phase = f/FRAMES
        t = TAU*phase
        controls, dx, dy, angle = motion(name, t)
        x = (xx-SIZE/2)/scale-dx
        y = (yy-SIZE/2)/scale-dy
        radians = math.radians(angle)
        cs, sn = math.cos(radians), math.sin(radians)
        sx = cs*x + sn*y + center[0]
        sy = -sn*x + cs*y + center[1]
        delta_x, delta_y = np.zeros_like(sx), np.zeros_like(sy)
        for (px, py), (vx, vy) in zip(points, controls):
            sigma = 39 if name in ('running', 'running-fast', 'dancing') else 32
            weight = np.exp(-((sx-px)**2+(sy-py)**2)/(2*sigma*sigma))
            delta_x += weight*vx
            delta_y += weight*vy
        x0, y0, x1, y1 = head
        # Smoothly pin the original crown and face, preserving identity.
        distances = np.minimum.reduce([sx-x0, x1-sx, sy-y0, y1-sy])
        pin = np.clip((distances+2)/9, 0, 1)
        sx -= delta_x*(1-pin)
        sy -= delta_y*(1-pin)
        mesh = []
        for row in range(32):
            for col in range(32):
                box = tuple(round(v) for v in (nodes[col], nodes[row], nodes[col+1], nodes[row+1]))
                quad = (sx[row,col], sy[row,col], sx[row+1,col], sy[row+1,col], sx[row+1,col+1], sy[row+1,col+1], sx[row,col+1], sy[row,col+1])
                mesh.append((box, quad))
        eye_art = blink(art, eyes, phase, name not in ('sad', 'crying', 'angry', 'shocked', 'in-love', 'in-love-pink', 'running-fast'))
        frame = eye_art.transform((SIZE, SIZE), Image.Transform.MESH, mesh, Image.Resampling.BICUBIC)
        frames.append(overlay(frame, name, phase))
    return frames


def gif_palette(frames):
    # One palette for every frame avoids flickering purple and facial details.
    width, height = frames[0].size
    samples = Image.new('RGB', (width*4, height*3), 'black')
    for n, f in enumerate(range(0, FRAMES, 4)):
        flat = Image.new('RGB', (width, height), 'black')
        flat.paste(frames[f], mask=frames[f].getchannel('A'))
        samples.paste(flat, ((n%4)*width, (n//4)*height))
    p = samples.quantize(colors=254, method=Image.Quantize.MEDIANCUT)
    palette = p.getpalette()[:762]+[0, 0, 0, 0, 0, 0]
    p.putpalette(palette)
    result = []
    for f in frames:
        rgb = f.convert('RGB')
        quant = rgb.quantize(palette=p, dither=Image.Dither.NONE)
        data = np.asarray(quant).copy()
        data[np.asarray(f.getchannel('A')) < 100] = 255
        q = Image.fromarray(data).convert('P')
        q.putpalette(palette)
        q.info['transparency'] = 255
        result.append(q)
    return result


def get_font(size):
    for p in ['/System/Library/Fonts/PingFang.ttc', '/System/Library/Fonts/STHeiti Light.ttc', '/Library/Fonts/Arial Unicode.ttf']:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def save_outputs(all_frames):
    for d in ['gif', 'webp', 'png']:
        (ROOT/d).mkdir(exist_ok=True)
    manifest = []
    for i, (pose, frames) in enumerate(zip(POSES, all_frames)):
        name, label = pose[:2]
        stem = f'{i+1:02d}-{name}'
        gif = gif_palette(frames)
        gif[0].save(ROOT/'gif'/f'{stem}.gif', save_all=True, append_images=gif[1:], duration=DURATION, loop=0, transparency=255, background=255, disposal=2, optimize=False)
        frames[0].save(ROOT/'webp'/f'{stem}.webp', save_all=True, append_images=frames[1:], duration=DURATION, loop=0, lossless=True, method=4)
        frames[0].save(ROOT/'png'/f'{stem}.png')
        manifest.append({'id': i+1, 'name': name, 'label': label, 'size': [SIZE, SIZE], 'frames': FRAMES, 'duration_ms': FRAMES*DURATION})
        print(f'Exported {stem}: {label}', flush=True)
    (ROOT/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')

    # A compact set of representative loops for inline preview.
    featured = [0, 2, 5, 7, 19, 21]
    preview_frames, contact_frames = [], []
    small_font, title_font = get_font(16), get_font(26)
    for f in range(FRAMES):
        preview = Image.new('RGB', (720, 500), '#111015')
        draw = ImageDraw.Draw(preview)
        for j, k in enumerate(featured):
            thumb = all_frames[k][f].resize((224, 224), Image.Resampling.LANCZOS)
            x, y = (j%3)*240+8, (j//3)*250
            preview.paste(thumb, (x, y), thumb)
            draw.text((x+112, y+224), POSES[k][1], font=small_font, fill='#d6c9ef', anchor='mt')
        preview_frames.append(preview)

        sheet = Image.new('RGB', (960, 1500), '#111015')
        d = ImageDraw.Draw(sheet)
        d.text((30, 22), 'KODEE · 22 个循环表情', font=title_font, fill='white')
        for j in range(22):
            x, y = (j%4)*240, (j//4)*235+67
            if j >= 20:
                x += 240
            thumb = all_frames[j][f].resize((216, 216), Image.Resampling.LANCZOS)
            sheet.paste(thumb, (x+12, y), thumb)
            d.text((x+120, y+208), f'{j+1:02d}  {POSES[j][1]}', font=small_font, fill='#c3b7da', anchor='mt')
        contact_frames.append(sheet)
    preview_frames[0].save(ROOT/'preview.gif', save_all=True, append_images=preview_frames[1:], duration=DURATION, loop=0, optimize=True)
    contact_frames[0].save(ROOT/'all-stickers.gif', save_all=True, append_images=contact_frames[1:], duration=DURATION, loop=0, optimize=True)
    contact_frames[0].save(ROOT/'all-stickers.jpg', quality=94)
    print('Preview sheets exported.', flush=True)


def main():
    all_frames = []
    for i, pose in enumerate(POSES):
        all_frames.append(render_pose(i, pose))
        print(f'Animated {i+1:02d} {pose[0]}', flush=True)
    save_outputs(all_frames)


if __name__ == '__main__':
    main()
