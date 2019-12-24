NVCC           = nvcc
NVCCFLAGS      = -g -G

SRCS = main.cu
INCS = vec3.h ray.h hittable.h hittable_list.h sphere.h  triangle.h helper.h camera.h cuda_helper.h material.h lambertian.h metal.h dielectric.h checker_board.h board.h

raytraicing: raytraicing.o
	$(NVCC) $(NVCCFLAGS) -o raytraicing raytraicing.o

raytraicing.o: $(SRCS) $(INCS)
	$(NVCC) $(NVCCFLAGS) -o raytraicing.o -c $(SRCS)

out.ppm: raytraicing
	rm -f out.ppm
	./raytraicing > out.ppm

out.jpg: out.ppm
	rm -f out.jpg
	ppmtojpeg out.ppm > out.jpg

clean:
	rm -f raytraicing raytraicing.o out.ppm out.jpg