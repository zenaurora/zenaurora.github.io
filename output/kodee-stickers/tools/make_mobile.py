"""Create a smaller 256px GIF set from the exported transparent WebPs."""
from PIL import Image, ImageSequence
from animate import ROOT, DURATION, gif_palette

destination = ROOT / 'mobile-gif'
destination.mkdir(exist_ok=True)
for path in sorted((ROOT / 'webp').glob('*.webp')):
    with Image.open(path) as im:
        frames = [f.convert('RGBA').resize((256, 256), Image.Resampling.LANCZOS)
                  for f in ImageSequence.Iterator(im)]
    frames = gif_palette(frames)
    frames[0].save(destination / f'{path.stem}.gif', save_all=True,
                   append_images=frames[1:], duration=DURATION, loop=0,
                   transparency=255, background=255, disposal=2, optimize=False)
    print(f'Created mobile GIF: {path.stem}', flush=True)
