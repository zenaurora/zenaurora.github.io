from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'source' / 'original.png'
NAMES = [
    'walking', 'happy', 'greeting', 'cute',
    'relaxing', 'running', 'running-fast', 'jumping',
    'jumping-playful', 'sitting', 'joyful', 'sad',
    'crying', 'holding', 'drawing', 'in-love',
    'in-love-pink', 'angry', 'shocked', 'dancing',
    'ice-skates', 'heart',
]

im = Image.open(SOURCE).convert('RGB')
# Crop labels out before extracting the original artwork.
rows = [(25, 299), (380, 660), (770, 1041), (1134, 1420), (1517, 1810), (1880, 2190)]
for i, name in enumerate(NAMES):
    row, col = divmod(i, 4) if i < 20 else (5, i - 19)
    left = col * 480
    top, bottom = rows[row]
    crop = im.crop((left + 24, top, left + 456, bottom))
    crop.thumbnail((360, 340), Image.Resampling.LANCZOS)
    tile = Image.new('RGB', (400, 400), 'black')
    tile.paste(crop, ((400 - crop.width) // 2, (400 - crop.height) // 2))
    tile.save(ROOT / 'source' / f'{i+1:02d}-{name}.png')
print(f'Prepared {len(NAMES)} character crops in {ROOT / "source"}')
