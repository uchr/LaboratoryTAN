constant float4 toXYZ[3] = {
	{0.412453f, 0.357580f, 0.180423f, 0.0f},
	{0.212671f, 0.715160f, 0.072169f, 0.0f},
	{0.019334f, 0.119193f, 0.950227f, 0.0f}
};

constant float4 refXYZ = { 0.95047f, 1.0f, 1.08883f, 1.0f };

// Функции для работы с цветом

// Converts from sRGB to CIE XYZ.
// The input values are assumed to be in [0.0f, 1.0f].
float4 rgb2xyz(float4 rgb) {
	float4 powRGB = pow((rgb+0.055f)/1.055f, 2.4f);
	float4 sclRGB = rgb / 12.92f;
	float4 resRGB = select(sclRGB, powRGB, rgb > 0.04045f);// * 100.0f; // [7.77, 100.00]
	float X = dot(rgb, toXYZ[0]); // max = 95.05
	float Y = dot(rgb, toXYZ[1]); // max = 100.0
	float Z = dot(rgb, toXYZ[2]); // max = 108.9
	return (float4)(X, Y, Z, 1.0f);
}

// Converts from CIE XYZ to CIE Lab.
float4 xyz2lab(float4 xyz) {
	float4 normXYZ = xyz / refXYZ; // max = 1.0
	float4 powXYZ = pow(normXYZ, 1.0f/3.0f); // max = 1.0
	float4 sclXYZ = normXYZ * 7.7870370370f + 16.0f/116.0f; //7.787f //24389.0f/3132.0f max < 1.0
	float4 resXYZ = select(sclXYZ, powXYZ, normXYZ > 0.0088564516f); //0.008856f //216.0f/24389.0f // max = 1.0
	float L = 116.0f * resXYZ.y - 16.0f; // [0, 100]
	float a = 500.0f * (resXYZ.x - resXYZ.y); // [-500, 500]
	float b = 200.0f * (resXYZ.y - resXYZ.z); // [-200, 200]
	return (float4)(L, a, b, 1.0f);
}

__kernel void rgb2lab(__read_only image2d_t src, __write_only image2d_t dst) {
	int x = get_global_id(0);
	int y = get_global_id(1);
	uint4 pixel = read_imageui(src, (int2)(x,y));
	float4 rgb = (float4) ((float) pixel.x / 255.0f, (float) pixel.y / 255.0f, (float) pixel.z / 255.0f, (float) pixel.w / 255.0f);
	float4 xyz = rgb2xyz(rgb);
	float4 lab = xyz2lab(xyz);
	write_imagef(dst, (int2)(x, y), lab);
}

__kernel void alphaSum(__read_only image2d_t image0, __read_only image2d_t image1, __write_only image2d_t dst, uchar alpha) {
	int2 pos = (int2) (get_global_id(0), get_global_id(1));
	uint4 pixel0 = read_imageui(image0, pos);
	uint4 pixel1 = read_imageui(image1, pos);
	float a = alpha / 255.0f;
	uint4 resultPixel = (uint4) (pixel0.x * a + pixel1.x * (1.0 - a), pixel0.y * a + pixel1.y * (1.0 - a), pixel0.z * a + pixel1.z * (1.0 - a), 255);
	write_imageui(dst, pos, resultPixel);
}

// Функции для отрисовки

__kernel void drawPolygons(__write_only image2d_t dst, __global const int *plgn, ushort numberOfPolygons, uchar numberOfVertices, uchar backgroundColor, uchar scale) {
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	uint4 pixel = (uint4) (backgroundColor, backgroundColor, backgroundColor, 255);
	const int stp = numberOfVertices*2+4;

	for (ushort i = 0; i < numberOfPolygons; ++i) {
		int result = 0;
		for (uchar j = 1; j <= numberOfVertices; ++j) {
			int prev = j - 1;
			int cur = j % numberOfVertices;
			int signLength = (scale*plgn[i*stp+cur*2+1]-scale*plgn[i*stp+prev*2+1])*(x-scale*plgn[i*stp+prev*2+0])
					-(scale*plgn[i*stp+cur*2+0]-scale*plgn[i*stp+prev*2+0])*(y-scale*plgn[i*stp+prev*2+1]);
			result += (signLength > 0) ? 1 :-1;
		}

		if (abs(result) == numberOfVertices) {
			float a = (float) plgn[i*stp+2*numberOfVertices+3] / 255.0f;
			pixel = (uint4)((1.0f - a) * pixel.x + a * plgn[i*stp+2*numberOfVertices+0], 
							(1.0f - a) * pixel.y + a * plgn[i*stp+2*numberOfVertices+1], 
							(1.0f - a) * pixel.z + a * plgn[i*stp+2*numberOfVertices+2], 
							255);
		}
	}

	write_imageui(dst, (int2) (x, y), (uint4) (pixel.x, pixel.y, pixel.z, 255));
}

__kernel void drawVoronoiL1(__write_only image2d_t dst, uint pointsNumber, __global const int *points, uint scale) {
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	int m = INT_MAX;
	int colorID = 0;
	for (uint i = 0; i < pointsNumber; ++i) {
		int dist = abs_diff((int)(points[i*6+0]*scale), x) + abs_diff((int)(points[i*6+1]*scale), y);
		if (dist < m) { m = dist; colorID = i; }
	}

	write_imageui(dst, (int2)(x, y), (uint4)(points[colorID*5+4], points[colorID*5+3], points[colorID*5+2], 255));
}

__kernel void drawVoronoiL2(__write_only image2d_t dst, uint pointsNumber, __global const int *points, uint scale) {
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	int m = INT_MAX;
	int colorID = 0;
	for (uint i = 0; i < pointsNumber; ++i) {
		int dist = sqrt((float)((points[i*5+0]*scale-x)*(points[i*5+0]*scale-x) + (points[i*5+1]*scale-y)*(points[i*5+1]*scale-y)));
		if (dist < m) { m = dist; colorID = i; }
	}

	write_imageui(dst, (int2)(x, y), (uint4)(points[colorID*5+4], points[colorID*5+3], points[colorID*5+2], 255));
}

__kernel void drawVoronoiLinf(__write_only image2d_t dst, uint pointsNumber, __global const int *points, uint scale) {
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	int m = INT_MAX;
	int colorID = 0;
	for (uint i = 0; i < pointsNumber; ++i) {
		int dist = (abs_diff((int)(points[i*6+0]*scale), x) > abs_diff((int)(points[i*6+1]*scale), y)) ? 
				abs_diff((int)(points[i*6+0]*scale), x) : abs_diff((int)(points[i*6+1]*scale), y);
		if (dist < m) { m = dist; colorID = i; }
	}

	write_imageui(dst, (int2)(x, y), (uint4)(points[colorID*5+4], points[colorID*5+3], points[colorID*5+2], 255));
}

__kernel void drawCircles(__write_only image2d_t dst, __global const int *circles, uchar numberOfCircles, uchar backgroundColor, uchar scale) {
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	uint4 pixel = (uint4) (backgroundColor, backgroundColor, backgroundColor, 255);
	const int stp = 7;

	for (ushort i = 0; i < numberOfCircles; ++i) {
		if ((x-circles[i*stp+0])*(x-circles[i*stp+0])+(y-circles[i*stp+1])*(y-circles[i*stp+1]) < circles[i*stp+2]*circles[i*stp+2]) {
			float a = ((float) circles[i*stp+3+3]) / 255.0f;
			pixel = (uint4)((1.0f - a) * pixel.x + a * circles[i*stp+3+0], 
							(1.0f - a) * pixel.y + a * circles[i*stp+3+1], 
							(1.0f - a) * pixel.z + a * circles[i*stp+3+2], 
							255);
		}
	}

	write_imageui(dst, (int2) (x, y), (uint4) (pixel.x, pixel.y, pixel.z, 255));
}

// Супер функция, которая рисует различные многоуольники и окружности
__kernel void draw(__write_only image2d_t dst, __global const int *smth, ushort numberOfSomething, uchar backgroundColor, uchar scale) {
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	uint4 pixel = (uint4) (backgroundColor, backgroundColor, backgroundColor, 255);
	int p = 0;

	bool flag;
	for (ushort i = 0; i < numberOfSomething; ++i) {
		flag = false;
		int result = 0;
		if (smth[p] > 0) { //Многоуольник
			uint numberOfVertices = smth[p];
			for (uchar j = 1; j <= numberOfVertices; ++j) {
				int prev = j - 1;
				int cur = j % numberOfVertices;
				int signLength = (scale*+smth[p+cur*2+1]-scale*smth[p+prev*2+1])*(x-scale*smth[p+prev*2+0])
						-(scale*smth[p+cur*2+0]-scale*smth[p+prev*2+0])*(y-scale*smth[p+prev*2+1]);
				result += (signLength > 0) ? 1 :-1;
			}
			if (abs(result) == numberOfVertices) flag = true;
			p += numberOfVertices*2;
		}
		else { // Окружность
			if ((x-smth[p+0])*(x-smth[p+0])+(y-smth[p+1])*(y-smth[p+1]) < smth[p+2]*smth[p+2]) flag = true;
			p += 3;
		}

		if (flag) {
			float a = (float) smth[p+3] / 255.0f;
			pixel = (uint4)((1.0f - a) * pixel.x + a * smth[p+0], 
							(1.0f - a) * pixel.y + a * smth[p+1], 
							(1.0f - a) * pixel.z + a * smth[p+2], 
							255);
		}
		p += 4;
	}

	write_imageui(dst, (int2) (x, y), (uint4) (pixel.x, pixel.y, pixel.z, 255));
}