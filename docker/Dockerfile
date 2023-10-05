FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

############################## SYSTEM PARAMETERS ##############################
# * Arguments
ARG USER=initial
ARG GROUP=initial
ARG UID=1000
ARG GID="${UID}"
ARG SHELL=/bin/bash
ARG HARDWARE=x86_64
ARG ENTRYPOINT_FILE=entrypoint.sh

# * Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all
# ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

# * Setup users and groups
RUN groupadd --gid "${GID}" "${GROUP}" \
    && useradd --gid "${GID}" --uid "${UID}" -ms "${SHELL}" "${USER}" \
    && mkdir -p /etc/sudoers.d \
    && echo "${USER}:x:${UID}:${UID}:${USER},,,:/home/${USER}:${SHELL}" >> /etc/passwd \
    && echo "${USER}:x:${UID}:" >> /etc/group \
    && echo "${USER} ALL=(ALL) NOPASSWD: ALL" > "/etc/sudoers.d/${USER}" \
    && chmod 0440 "/etc/sudoers.d/${USER}"

# * Replace apt urls
# ? Change to tku
RUN sed -i 's@archive.ubuntu.com@ftp.tku.edu.tw@g' /etc/apt/sources.list
# ? Change to Taiwan
# RUN sed -i 's@archive.ubuntu.com@tw.archive.ubuntu.com@g' /etc/apt/sources.list

# * Time zone
ENV TZ=Asia/Taipei
RUN ln -snf /usr/share/zoneinfo/"${TZ}" /etc/localtime && echo "${TZ}" > /etc/timezone

# * Copy custom configuration
# ? Requires docker version >= 17.09
COPY --chmod=0775 ./${ENTRYPOINT_FILE} /entrypoint.sh
COPY --chown="${USER}":"${GROUP}" --chmod=0775 config config
# ? docker version < 17.09
# COPY ./${ENTRYPOINT_FILE} /entrypoint.sh
# COPY config config
# RUN sudo chmod 0775 /entrypoint.sh && \
    # sudo chown -R "${USER}":"${GROUP}" config \
    # && sudo chmod -R 0775 config

############################### INSTALL #######################################
# * Install packages
RUN apt update
RUN apt install -y --no-install-recommends \
        sudo \
        vim \
        git \
        htop \
        cmake \
        wget \
        curl \
        psmisc \
        udev \
        # * Shell
        tmux \
        terminator \
        # * base tools
        python3-pip \
        python3-dev \
        python3-setuptools \
        # * Work tools
        usbutils \
        python3-pyqt5 \
        # * ROS depend tools
        lsb-release \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

RUN ./config/pip/pip_setup.sh

# * Install Pytorch, MMCV
# [!!!] https://zhuanlan.zhihu.com/p/408209150
# 安裝順序: python – pytorch – mmcv -mmdet 

#pytorch：https://pytorch.org/get-started/previous-versions
#mmcv：https://mmcv.readthedocs.io/en/latest/get_started/installation.html 

RUN pip3 install --upgrade --force-reinstall pip && \
    pip3 install \
    torch==1.10.1+cu111 \
    torchvision==0.11.2+cu111 \
    torchaudio==0.10.1 \
    -f https://download.pytorch.org/whl/cu111/torch_stable.html

# SOLO depend tools
RUN pip3 install \
    terminaltables \
    pycocotools \
    shapely \
    scipy \
    -U openmim 
RUN mim install mmcv-full==1.7.1
RUN mim install mmengine
RUN pip3 install yapf==0.40.1


# * Install ROS noetic
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > \
    /etc/apt/sources.list.d/ros-latest.list \
    && apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' \
    --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 \
    && curl -sSL \
    'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC1CF6E31E6BADE8868B172B4F42ED6FBAB17C654' \
    | sudo apt-key add - \
    && apt update && apt install -y --no-install-recommends ros-noetic-desktop-full \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV ROS_DISTRO noetic
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >>/home/$USER/.bashrc

RUN apt update \
    && apt install -y --no-install-recommends \
        # * ROS depend tools
        ros-noetic-ddynamic-reconfigure \
        ros-noetic-rgbd-launch \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

RUN sudo apt update
RUN sudo apt install -y software-properties-common 

RUN sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE && sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
####Realsense####
RUN sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u
## Install packages
RUN apt-get update && apt-get install -y --no-install-recommends \
  #Realsense SDK depend
#   librealsense2-dkms \
  librealsense2-utils \
  librealsense2-dev \
  librealsense2-dbg \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists  

WORKDIR /

############################## USER CONFIG ####################################
# * Switch user to ${USER}
USER ${USER}

RUN ./config/shell/bash_setup.sh "${USER}" "${GROUP}" \
    && ./config/shell/terminator/terminator_setup.sh "${USER}" "${GROUP}" \
    && ./config/shell/tmux/tmux_setup.sh "${USER}" "${GROUP}" \
    && sudo rm -rf /config

# # * Create work directory
# RUN sudo mkdir -p /home/"${USER}"/work/Altek_3D_Camera

# * Install Altek SDK
# # ? COPY workable Altek_3D_Camera SDK (v2.49.0)
# RUN sudo mkdir -p /home/"${USER}"/work/Altek_3D_Camera
# # WORKDIR /home/"${USER}"/work/
# COPY ./Altek_3D_Camera /home/"${USER}"/work/Altek_3D_Camera

# # ? Altek_3D_Camera Github (v2.49.0 tag:v67) currently CANNOT detect Altek Camera
# RUN sudo git clone https://github.com/owin1022/Altek_3D_Camera.git /home/"${USER}"/work/Altek_3D_Camera

# WORKDIR /home/"${USER}"/work/Altek_3D_Camera
# RUN sudo mkdir build 
# WORKDIR /home/"${USER}"/work/Altek_3D_Camera/build
# RUN sudo cmake .. -DCMAKE_BUILD_TYPE=Release 
# RUN sudo make uninstall && \
#     sudo make clean && \
#     sudo make -j12 && \
#     sudo make install

# * Switch workspace to ~/work
RUN sudo mkdir -p /home/"${USER}"/work
WORKDIR /home/"${USER}"/work

# WORKDIR /home/"${USER}"/work/

# * Make SSH available
EXPOSE 22

ENTRYPOINT [ "/entrypoint.sh", "terminator" ]
# ENTRYPOINT [ "/entrypoint.sh", "tmux" ]
# ENTRYPOINT [ "/entrypoint.sh", "bash" ]
# ENTRYPOINT [ "/entrypoint.sh" ]
