import cv2
from pathlib import Path

spin_path = Path('/scratch/sjc-advdev01/ml/datasets/speedplusv2/spin')
image_path = spin_path / 'images'
output_path = spin_path / 'images_768x512_GRAY'

png_files = list(image_path.glob('*.png'))

#dir_path = Path('/path/to/directory')
for n,filepath in enumerate(png_files):
    if n%100 ==0:
        print('file '+str(n)+' of '+str(len(png_files)))
    image    = cv2.imread(filepath, cv2.IMREAD_COLOR)
    image    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # BGR2RGB
    image    = cv2.resize(image, (768,512))
    cv2.imwrite(output_path / filepath.name, image)
