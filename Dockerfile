FROM python:3-alpine

RUN apk --no-cache add git mesa-osmesa mesa-gles gcc gfortran python-dev build-base wget freetype-dev fontconfig-dev libpng-dev libjpeg-turbo-dev openblas-dev && pip install numpy vispy Pillow

COPY . /blockcrafter
RUN cd /blockcrafter && pip wheel .


FROM python:3-alpine

RUN apk --no-cache add mesa-osmesa mesa-gles libpng freetype fontconfig-dev libjpeg-turbo openblas binutils

COPY --from=0 /blockcrafter/*.whl /blockcrafter/
RUN rm /blockcrafter/*manylinux1*.whl && pip install /blockcrafter/*.whl

ENV VISPY_GL_LIB /usr/lib/libGLESv2.so.2
ENV OSMESA_LIBRARY /usr/lib/libOSMesa.so.8

ENTRYPOINT ["/usr/local/bin/blockcrafter-export", "--osmesa"]
