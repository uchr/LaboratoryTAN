import numpy as np
import pyopencl as cl

import argparse
from skimage import io
from scipy.misc import imresize

import time

# Инициализация OpenCL
#===========================
print(cl.get_platforms())
platform = cl.get_platforms()[1]
device = platform.get_devices()[0]
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags

uint8Format = cl.ImageFormat(cl.channel_order.BGRA, cl.channel_type.UNSIGNED_INT8)
floatFormat = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)

with open('KernelsDrawing.cl') as kernelFile:
	kernel = kernelFile.read()

prg = cl.Program(ctx, kernel).build()
#===========================

# Преобразует строку аргументов с словарь
def getArgs():
	parser = argparse.ArgumentParser(description='Абстрактно картиночный генератор')
	parser.add_argument("IMAGE_NAME", type=str, help="Целевое изображение")
	parser.add_argument('-p', '--polygons', dest='NUMBER_OF_POLYGONS', type=int, default=50, help='Количество полигонов')
	parser.add_argument('-v', '--vertices', dest='NUMBER_OF_VERTICES', type=int, default=3, help='Количество вершин')
	parser.add_argument('-g', '--genType', dest='GEN_TYPE', type=int, default=1, help='Тип изображения')
	parser.add_argument('--minAlpha', dest='MIN_ALPHA', type=int, default=100, help='Минимальная прозрачность для многоугольника')
	parser.add_argument('--pSize', dest='POLYGON_SIZE', type=int, default=25, help='Размер многоугольника')
	parser.add_argument('--population', dest='POPULATION', type=int, default=120, help='Размер популяции')
	parser.add_argument('--freq', dest='FREQ', type=int, default=25, help='Частота сохранения изображений')
	parser.add_argument('--numGen', dest='NGEN', type=int, default=10000, help='Количество полоклений')
	parser.add_argument('--backgroundColor', dest='BACKGROUND_COLOR', type=int, default=255, help='Цвет фона от черного к белому')
	parser.add_argument('-n', '--new', dest='NEW', action='store_true', help='Начать генерацию заново')

	return parser.parse_args()

# Загружает целевое изображение в формате RGB
def loadImageRGB(path):
	PIC = io.imread(path)
	if PIC.shape[2] == 3:
		PIC = rgb2rgba(PIC)
	return PIC, PIC.shape[1::-1]

# Загружает целевое изображение в формате Lab
def loadImageLab(path):
	PIC = io.imread(path)
	temp = rgb2lab(PIC, PIC.shape[1::-1])
	return temp, temp.shape[1::-1]

# Сохраняет изображение в формате RGB
def saveImageRGB(path, image):
	#TODO Добавить проверку на существование директории
	io.imsave(path, image)

# Добавление альфа канала
def rgb2rgba(rgb):
	rgba = np.empty((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
	rgba[:, :, 0] = rgb[:, :, 0]
	rgba[:, :, 1] = rgb[:, :, 1]
	rgba[:, :, 2] = rgb[:, :, 2]
	rgba[:, :, 3] = 255
	return rgba

def appendImage(image0, image1):
	temp = np.zeros((image0.shape[0], image0.shape[1] + image1.shape[1]), dtype=image0.dtype.type)
	temp[:,:image0.shape[1]] = image0[:,:]
	temp[:,image0.shape[1]:] = image1[:,:]

# Конвертирование из RGB в Lab
def rgb2lab(image, shape):
	if image.shape[2] == 3: # Check alpha channel
		srcUintImage = cl.Image(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR | mf.ALLOC_HOST_PTR, uint8Format, shape=shape, hostbuf=rgb2rgba(image))
		cl.enqueue_copy(queue, srcUintImage, rgb2rgba(image), origin=(0, 0), region=shape).wait()
	else:
		srcUintImage = cl.Image(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR | mf.ALLOC_HOST_PTR, uint8Format, shape=shape, hostbuf=image)
		cl.enqueue_copy(queue, srcUintImage, image, origin=(0, 0), region=shape).wait()
	dstFloatImage = cl.Image(ctx, mf.WRITE_ONLY, floatFormat, shape=shape)

	prg.rgb2lab(queue, shape, None, srcUintImage, dstFloatImage)
	temp = np.zeros(shape[1::-1] + (4,), dtype=np.float32)
	cl.enqueue_copy(queue, temp, dstFloatImage, origin=(0, 0), region=shape).wait()

	dstFloatImage.release()
	srcUintImage.release()
	return temp

# Отрисовка полигонов в RGB
def polygons2rgb(polygons, numberOfVertices, shape, backgroundColor, scale = 1):
	scaleShape = (int(shape[0] * scale), int(shape[1] * scale))

	polygonsBuffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.array(polygons, dtype=np.int32).ravel())
	dstUintImage = cl.Image(ctx, mf.READ_WRITE, uint8Format, shape=scaleShape)

	temp = np.zeros(scaleShape[1::-1] + (4,), dtype=np.uint8)
	prg.drawPolygons(queue, scaleShape, None, dstUintImage, polygonsBuffer, np.uint16(len(polygons)), np.uint8(numberOfVertices), np.uint8(backgroundColor), np.uint8(scale))
	cl.enqueue_copy(queue, temp, dstUintImage, origin=(0, 0), region=scaleShape).wait()

	dstUintImage.release()
	polygonsBuffer.release()
	return temp

# Отрисовка полигонов в RGB со сглаживанием
def polygons2rgbAA(polygons, numberOfVertices, shape, backgroundColor):
	return imresize(polygons2rgb(polygons, numberOfVertices, shape, backgroundColor, scale = 4), (shape[1] * 2, shape[0] * 2))

# Отрисовка полигонов в Lab
def polygons2lab(polygons, numberOfVertices, shape, backgroundColor):
	dstFloatImage = cl.Image(ctx, mf.WRITE_ONLY, floatFormat, shape=shape)
	dstUintImage = cl.Image(ctx, mf.READ_WRITE, uint8Format, shape=shape)

	temp = np.zeros(shape[1::-1] + (4,), dtype=np.float32)
	polygonsBuffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.array(polygons, dtype=np.int32).ravel())
	prg.drawPolygons(queue, shape, None, dstUintImage, polygonsBuffer, np.uint16(len(polygons)), np.uint8(numberOfVertices), np.uint8(backgroundColor), np.uint8(1))
	prg.rgb2lab(queue, shape, None, dstUintImage, dstFloatImage)
	cl.enqueue_copy(queue, temp, dstFloatImage, origin=(0, 0), region=shape).wait()

	dstUintImage.release()
	dstFloatImage.release()
	return temp

# Отрисовка диаграмм Вороного в RGB
# Distance from the {"L1", "L2", "Linf"}
def points2rgb(points, shape, distance = "L2", scale = 1):
	scaleShape = (int(shape[0] * scale), int(shape[1] * scale))

	pointsBuffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.array(points, dtype=np.int32).ravel())
	dstUintImage = cl.Image(ctx, mf.READ_WRITE, uint8Format, shape=scaleShape)

	temp = np.zeros(scaleShape[1::-1] + (4,), dtype=np.uint8)
	if distance == "L1":
		prg.drawVoronoiL1(queue, scaleShape, None, dstUintImage, np.uint32(len(points)), pointsBuffer, np.uint32(scale))
	elif distance == "L2":
		prg.drawVoronoiL2(queue, scaleShape, None, dstUintImage, np.uint32(len(points)), pointsBuffer, np.uint32(scale))
	elif distance == "Linf":
		prg.drawVoronoiLinf(queue, scaleShape, None, dstUintImage, np.uint32(len(points)), pointsBuffer, np.uint32(scale))
	cl.enqueue_copy(queue, temp, dstUintImage, origin=(0, 0), region=scaleShape)

	dstUintImage.release()
	pointsBuffer.release()
	return temp

# Отрисовка диаграмм Вороного в RGB со сглаживанием
# Distance from the {"L1", "L2", "Linf"}
def points2rgbAA(points, shape, distance = "L2"):
	return imresize(points2rgb(points, shape, distance = distance, scale = 4), (shape[1] * 2, shape[0] * 2))

# Отрисовка диаграмм Вороного в Lab
# Distance from the {"L1", "L2", "Linf"}
def points2lab(points, shape, distance = "L2"):
	dstFloatImage = cl.Image(ctx, mf.WRITE_ONLY, floatFormat, shape=shape)
	dstUintImage = cl.Image(ctx, mf.READ_WRITE, uint8Format, shape=shape)

	temp = np.zeros(shape[1::-1] + (4,), dtype=np.float32)
	pointsBuffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.array(points, dtype=np.int32).ravel())
	if distance == "L1":
		prg.drawVoronoiL1(queue, shape, None, dstUintImage, np.uint32(len(points)), pointsBuffer, np.uint32(1))
	elif distance == "L2":
		prg.drawVoronoiL2(queue, shape, None, dstUintImage, np.uint32(len(points)), pointsBuffer, np.uint32(1))
	elif distance == "Linf":
		prg.drawVoronoiLinf(queue, shape, None, dstUintImage, np.uint32(len(points)), pointsBuffer, np.uint32(1))
	prg.rgb2lab(queue, shape, None, dstUintImage, dstFloatImage)
	cl.enqueue_copy(queue, temp, dstFloatImage, origin=(0, 0), region=shape)

	dstUintImage.release()
	dstFloatImage.release()
	return temp

# Отрисовка полигонов в RGB
def circles2rgb(circles, shape, backgroundColor, scale = 1):
	scaleShape = (int(shape[0] * scale), int(shape[1] * scale))

	circlesBuffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.array(circles, dtype=np.int32).ravel())
	dstUintImage = cl.Image(ctx, mf.READ_WRITE, uint8Format, shape=scaleShape)

	temp = np.zeros(scaleShape[1::-1] + (4,), dtype=np.uint8)
	prg.drawCircles(queue, scaleShape, None, dstUintImage, circlesBuffer, np.uint8(len(circles)), np.uint8(backgroundColor), np.uint8(scale))
	cl.enqueue_copy(queue, temp, dstUintImage, origin=(0, 0), region=scaleShape).wait()

	dstUintImage.release()
	circlesBuffer.release()
	return temp

# Отрисовка полигонов в RGB со сглаживанием
def circles2rgbAA(circles, shape, backgroundColor):
	return imresize(circles2rgb(circles, shape, backgroundColor, 4), (int(shape[1] * 2), int(shape[0] * 2)))

# Отрисовка полигонов в Lab
def circles2lab(circles, shape, backgroundColor):
	dstFloatImage = cl.Image(ctx, mf.WRITE_ONLY, floatFormat, shape=shape)
	dstUintImage = cl.Image(ctx, mf.READ_WRITE, uint8Format, shape=shape)

	temp = np.zeros(shape[1::-1] + (4,), dtype=np.float32)
	circlesBuffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.array(circles, dtype=np.int32).ravel())
	prg.drawCircles(queue, shape, None, dstUintImage, circlesBuffer, np.uint8(len(circles)), np.uint8(backgroundColor), np.uint8(1))
	prg.rgb2lab(queue, shape, None, dstUintImage, dstFloatImage)
	cl.enqueue_copy(queue, temp, dstFloatImage, origin=(0, 0), region=shape).wait()

	dstUintImage.release()
	dstFloatImage.release()
	circlesBuffer.release()
	return temp
