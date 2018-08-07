#FROM python:3-slim-stretch
FROM python:3-alpine

#RUN apt-get update && apt-get install -y libgl1-mesa-glx libosmesa-dev && rm -rf /var/lib/apt/lists/*

RUN apk --no-cache add mesa-osmesa mesa-gles gcc gfortran python-dev build-base wget freetype-dev fontconfig-dev libpng-dev libjpeg-turbo-dev openblas-dev && \
    pip --no-cache-dir install vispy Pillow && \
    apk del python-dev gcc openblas-dev gfortran build-base wget && apk --no-cache add openblas binutils && rm -rf /var/cache/apk/*

COPY . /blockcrafter

ENV VISPY_GL_LIB /usr/lib/libGLESv2.so.2
ENV OSMESA_LIBRARY /usr/lib/libOSMesa.so.8

ENTRYPOINT ["/blockcrafter/export.py"]
