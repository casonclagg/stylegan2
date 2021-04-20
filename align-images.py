from pathlib import Path

orig_img_path = Path('imgs')

aligned_imgs_path = Path('aligned_imgs')
if not aligned_imgs_path.exists():
    aligned_imgs_path.mkdir()
import dlib
from align_face import align_face

# Align all of our images using a landmark detection model!
all_imgs = list(orig_img_path.iterdir())
for img in all_imgs:
    print(f'aligning {img}')
    try:
        align_face(str(img)).save(aligned_imgs_path/('aligned_'+img.name))
    except:
        print('no face')


# aligned_img_set = list(aligned_imgs_path.iterdir())
# aligned_img_set.sort()
# aligned_img_set = [Image.open(x) for x in aligned_img_set]

# orig_img_set = list(orig_img_path.iterdir())
# orig_img_set.sort()
# orig_img_set = [Image.open(x) for x in orig_img_set]