FROM python:alpine3.12

RUN apk --no-cache add git py3-pip mesa-osmesa mesa-gles gcc gfortran python3-dev build-base wget freetype-dev fontconfig-dev libpng-dev libjpeg-turbo-dev openblas-dev
RUN pip3 install -U scikit-build make
RUN pip3 install numpy==1.17.5
RUN pip3 install Pillow==6.2.2
RUN pip3 install vispy==0.5.3

COPY . /blockcrafter
RUN cd /blockcrafter && pip wheel .


FROM python:3-alpine

RUN apk --no-cache add mesa-osmesa mesa-gles libpng freetype fontconfig-dev libjpeg-turbo openblas binutils shadow

COPY --from=0 /blockcrafter/*.whl /blockcrafter/
RUN rm -f /blockcrafter/*manylinux1*.whl && pip install /blockcrafter/*.whl

COPY entrypoint.sh /

ENV VISPY_GL_LIB /usr/lib/libGLESv2.so.2
ENV OSMESA_LIBRARY /usr/lib/libOSMesa.so.8

ENTRYPOINT ["/entrypoint.sh"]
