FROM fairembodied/habitat-challenge:testing_2022_habitat_base_docker

# Install dependencies
RUN /bin/bash -c "apt-get update; apt-get install -y ffmpeg"
RUN conda install cudatoolkit=11.1 -c conda-forge
ADD scripts/env_setup.sh /setup.sh
RUN /bin/bash /setup.sh

ADD scripts/ /scripts/
ADD scripts/submission.sh /submission.sh
ADD configs/challenge_objectnav2022.local.rgbd.yaml /challenge_objectnav2022.local.rgbd.yaml

ENV AGENT_EVALUATION_TYPE local
ENV IS_LOCAL true
ENV PYTHONPATH "${PYTHONPATH}:/Kemono"
ENV TRACK_CONFIG_FILE "/challenge_objectnav2022.local.rgbd.yaml"

CMD ["/bin/bash", "-c", "source activate habitat && export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && bash submission.sh"]

