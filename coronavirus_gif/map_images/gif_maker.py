import imageio
from pathlib import Path
from pygifsicle import optimize




path = Path('map_images')
images= list(path.glob('*.png'))
images.sort()
list_images =[]
for image in images:
	list_images.append(imageio.imread(image))

imageio.mimwrite('coronavirus_evolve_map.gif',list_images, loop=1, fps=2)


gif_path = 'coronavirus_evolve_map.gif'
optimize(gif_path, 'coronavirus_evolve_map_optimized.gif')
optimize(gif_path)