FROM ros:noetic-ros-core-focal

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    python3-pip \
    python3-rosdep \
    python3-rosinstall \
    python3-vcstools \
    && rm -rf /var/lib/apt/lists/*

# bootstrap rosdep
RUN rosdep init && \
  rosdep update --rosdistro $ROS_DISTRO

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-base=1.5.0-1* \
    ros-noetic-cv-bridge \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# python libraries for cinaps_vlm module
RUN pip3 install --no-cache-dir \
    anthropic==0.36.2 \
    openai==1.51.0 \
    pillow==10.4.0 \
    torch==2.4.0 \
    transformers==4.44.2

# Source ROS
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
