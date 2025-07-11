# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains the CMU Vision-Language-Autonomy Challenge codebase, which integrates computer vision and natural language understanding for robot navigation. The challenge provides a robot system equipped with a 3D lidar and 360 camera to navigate environments based on natural language queries.

## Key Commands

### Building the AI Module
```bash
cd ai_module/
catkin_make
```

### Building the System (Unity Simulator)
```bash
cd system/unity/
catkin_make
```

### Running the Complete System
1. Launch the system simulator (in one terminal):
```bash
./launch_system.sh
```

2. Launch the AI module (in another terminal):
```bash
./launch_module.sh
```

### Docker Commands
For systems without GPU:
```bash
docker compose -f compose.yml up --build -d
```

For systems with GPU:
```bash
docker compose -f compose_gpu.yml up --build -d
```

Access containers:
```bash
docker exec -it ubuntu20_ros_system bash
docker exec -it ubuntu20_ros bash
```

## Architecture Overview

### Repository Structure
- **ai_module/**: Contains the vision-language model that teams develop
  - Currently has a dummy VLM implementation in C++ that demonstrates the required interfaces
  - Teams replace this with their own implementation
  
- **system/**: Contains the base navigation system
  - **unity/**: Unity-based simulator and ROS integration
  - **matterport/**: Alternative Matterport3D-based simulator
  
- **docker/**: Docker configuration files for development environment
- **questions/**: Challenge questions and ground truth data for 15 training scenes

### Key ROS Topics

**Inputs to AI Module:**
- `/challenge_question` (std_msgs/String): Natural language query
- `/camera/image` (Image): 360° camera feed
- `/registered_scan` (PointCloud2): Registered 3D lidar data
- `/sensor_scan` (PointCloud2): Raw 3D lidar data
- `/terrain_map` (PointCloud2): Local terrain analysis
- `/state_estimation` (Odometry): Vehicle pose
- `/traversable_area` (PointCloud2): Navigable area map
- `/object_markers` (MarkerArray): Ground-truth semantics (training only)

**Outputs from AI Module:**
- `/numerical_response` (Int32): For "how many" questions
- `/selected_object_marker` (Marker): For "find the" object reference questions
- `/way_point_with_heading` (Pose2D): For navigation instructions

### Question Types
1. **Numerical**: Count objects matching criteria, return integer
2. **Object Reference**: Find specific object, return bounding box marker
3. **Instruction-Following**: Navigate path with constraints, return waypoint sequence

### Development Workflow

1. **Modify AI Module**: Replace dummy_vlm with your implementation
2. **Test Locally**: Use provided Unity scenes for development
3. **Build Docker Image**: Update Dockerfile if needed for dependencies
4. **Submit**: Push to Docker Hub and submit GitHub repository link

### Environment Setup Requirements
- Ubuntu 20.04
- ROS Noetic
- Docker (with nvidia-container-toolkit for GPU support)
- Unity environment models (download separately)

### Evaluation Details
- 10-minute time limit per question (exploration + answering)
- System relaunches for each question (no retained state)
- Scored on accuracy and timing
- Test on 3 held-out Unity scenes with 5 questions each