FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime as base
WORKDIR /workdir
COPY recsiam ./recsiam
COPY scripts ./scripts
ENV PYTHONPATH /workdir:$PYTHONPATH
RUN pip install lz4==2.1 scikit-learn==0.21 scikit-image==0.15 pytorch-ignite


FROM base as basepreprocess
RUN apt-get update && apt-get install -y unzip &&\
    python -c "import torchvision.models as models ; models.resnet152(pretrained=True)"


FROM basepreprocess as reid100preprocess
RUN curl https://ndownloader.figshare.com/files/17435471 --output dataset.zip && \
    unzip -q dataset.zip  && \
    rm dataset.zip &&\
    mkdir /reid100
ARG gpu=""
RUN if [ "$gpu" = "y" ] ; then export cudaflag="-c" ; fi ;\
    python scripts/pre_embed.py -z ${cudaflag} --cnn-embedding resnet152 -b 1 \
           dataset/reid100 /reid100 ;\
    python scripts/pre_embed.py -z ${cudaflag} --cnn-embedding resnet152 -b 1 \
           dataset/mugs9 /reid9mugs
RUN python scripts/fs2desc.py /reid100 descriptor.json
RUN python scripts/fs2desc.py /reid9mugs mugs9descriptor.json


FROM basepreprocess as corepreprocess
RUN curl http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip --output coredataset.zip && \
    unzip -q coredataset.zip  && \
    rm coredataset.zip &&\
    mkdir /core
ARG gpu=""
RUN if [ "$gpu" = "y" ] ; then export cudaflag="-c" ; fi ;\
    python scripts/pre_embed.py -z ${cudaflag} --cnn-embedding resnet152 -b 1 \
           core50_128x128/ /core &&\
    rm -r core50_128x128/
RUN python scripts/coredesc.py /core coredescriptor.json


FROM base as experiment
COPY --from=reid100preprocess /reid100 /reid100
COPY --from=reid100preprocess /workdir/descriptor.json /reid100.json
COPY --from=reid100preprocess /reid9mugs /reid9mugs
COPY --from=reid100preprocess /workdir/mugs9descriptor.json /reid9mugs.json
COPY --from=corepreprocess /core /core
COPY --from=corepreprocess /workdir/coredescriptor.json /core.json
COPY runexp.sh runexp.sh
VOLUME ["/results", "/inputs", "/outputs", "/info"]
CMD bash runexp.sh 
