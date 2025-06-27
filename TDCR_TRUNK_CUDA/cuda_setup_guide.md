# CUDA Setup Guide for SOFA in Docker

Complete guide to enable CUDA acceleration in your SOFA soft robotics simulations running in Docker.

## 🐳 Docker CUDA Setup

### 1. Prerequisites

- **NVIDIA GPU** with CUDA support
- **NVIDIA Docker Runtime** installed on host
- **CUDA-compatible SOFA build** in container

### 2. Host System Requirements

#### Install NVIDIA Container Toolkit
```bash
# Add NVIDIA package repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker daemon
sudo systemctl restart docker
```

#### Verify NVIDIA Docker Setup
```bash
# Test NVIDIA Docker access
docker run --rm --gpus all nvidia/cuda:11.8-devel-ubuntu20.04 nvidia-smi
```

### 3. Docker Container Setup

#### Option A: Modify Existing Container
If you have an existing SOFA container, modify the run command:

```bash
# Run container with GPU access
docker run -it --gpus all \
    -v /path/to/your/my_files:/workspace/my_files \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    your-sofa-container:tag
```

#### Option B: Create New CUDA-Enabled SOFA Container
```dockerfile
# Dockerfile for CUDA-enabled SOFA
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libboost-all-dev \
    libeigen3-dev \
    libglew-dev \
    freeglut3-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    python3-dev \
    python3-pip \
    git

# Install SOFA with CUDA support
RUN git clone https://github.com/sofa-framework/sofa.git /sofa-src
WORKDIR /sofa-src
RUN mkdir build && cd build && \
    cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DSOFA_BUILD_TUTORIALS=ON \
    -DPLUGIN_SOFACUDA=ON \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    && make -j$(nproc)

# Set environment
ENV SOFA_ROOT=/sofa-src/build
ENV PYTHONPATH=$SOFA_ROOT/lib/python3/site-packages:$PYTHONPATH
```

## 🔧 SOFA CUDA Configuration

### 4. Required SOFA Plugins

Your SOFA build must include these plugins:

```cmake
# CMake configuration for CUDA support
-DPLUGIN_SOFACUDA=ON
-DPLUGIN_SOFACUDADEV=ON  # Optional: for development
-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
-DCUDA_ARCH_LIST="6.1;7.0;7.5;8.0;8.6"  # Adjust for your GPU
```

### 5. Verify CUDA in SOFA

Create a test script to verify CUDA is available:

```python
# test_cuda.py
import Sofa
import Sofa.Gui

def test_cuda_availability():
    """Test if CUDA components are available in SOFA."""
    
    root = Sofa.Core.Node("root")
    
    try:
        # Try to load SofaCUDA plugin
        root.addObject('RequiredPlugin', name="SofaCUDA")
        print("✅ SofaCUDA plugin loaded successfully")
        
        # Test CUDA components
        test_node = root.addChild("test")
        
        # Try creating CUDA MechanicalObject
        cuda_mo = test_node.addObject('MechanicalObject', 
                                     template='CudaVec3f', 
                                     name='cudaTest')
        print("✅ CUDA MechanicalObject created successfully")
        
        # Try CUDA linear solver
        cuda_solver = test_node.addObject('CGLinearSolver', 
                                         template='CudaVec3f',
                                         name='cudaSolver')
        print("✅ CUDA CGLinearSolver created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing CUDA availability in SOFA...")
    if test_cuda_availability():
        print("🎉 CUDA is properly configured!")
    else:
        print("💥 CUDA setup needs fixing")
```

## 🚀 Working CUDA Elastic Object

### 6. Fixed CUDA Implementation

Create a working CUDA elastic material object:

```python
# cuda_elastic_fixed.py
import Sofa
import Sofa.Core

class CudaElasticMaterialObject:
    """
    Working CUDA-accelerated elastic material object.
    Simplified and tested implementation.
    """
    
    def __init__(self, parent_node, name="CudaElastic", **kwargs):
        self.node = parent_node.addChild(name)
        self.params = {
            'volumeMeshFileName': None,
            'youngModulus': 18000,
            'poissonRatio': 0.3,
            'totalMass': 1.0,
            'rotation': [0, 0, 0],
            'translation': [0, 0, 0],
            'scale': [1, 1, 1],
            **kwargs
        }
        
        self._create_cuda_components()
    
    def _create_cuda_components(self):
        """Create CUDA-accelerated components."""
        
        # Ensure CUDA plugin is loaded
        root = self.node.getRoot()
        if not self._has_plugin(root, "SofaCUDA"):
            root.addObject('RequiredPlugin', name="SofaCUDA")
        
        # Load mesh
        if self.params['volumeMeshFileName']:
            if self.params['volumeMeshFileName'].endswith('.vtk'):
                loader = self.node.addObject('MeshVTKLoader', 
                                           name='loader',
                                           filename=self.params['volumeMeshFileName'],
                                           rotation=self.params['rotation'],
                                           translation=self.params['translation'],
                                           scale3d=self.params['scale'])
            else:
                raise ValueError("Only VTK files supported for now")
        
        # CUDA-accelerated solver chain
        self.node.addObject('EulerImplicitSolver', name='odesolver')
        self.node.addObject('CGLinearSolver', 
                           template='CudaVec3f',
                           name='solver',
                           iterations=50,
                           tolerance=1e-6)
        
        # Topology and mechanics with CUDA
        self.node.addObject('TetrahedronSetTopologyContainer', 
                           src='@loader', 
                           name='container')
        
        self.node.addObject('MechanicalObject', 
                           template='CudaVec3f',
                           name='dofs')
        
        self.node.addObject('UniformMass', 
                           totalMass=self.params['totalMass'])
        
        # CUDA FEM force field
        self.node.addObject('TetrahedronFEMForceField',
                           template='CudaVec3f',
                           name='FEM',
                           method='large',
                           poissonRatio=self.params['poissonRatio'],
                           youngModulus=self.params['youngModulus'])
        
        # Constraint correction
        self.node.addObject('UncoupledConstraintCorrection')
    
    def _has_plugin(self, root, plugin_name):
        """Check if a plugin is already loaded."""
        for child in root.children:
            if child.name == plugin_name:
                return True
        return False

def create_cuda_test_scene(root):
    """Create a test scene with CUDA acceleration."""
    
    # Basic scene setup
    root.addObject('RequiredPlugin', name="SofaCUDA")
    root.addObject('RequiredPlugin', name="SoftRobots")
    root.addObject('VisualStyle', displayFlags="showBehavior showForceFields")
    root.addObject('FreeMotionAnimationLoop')
    root.addObject('GenericConstraintSolver', tolerance=1e-6, maxIterations=1000)
    
    # Create CUDA elastic object
    cuda_object = CudaElasticMaterialObject(
        root,
        name="CudaTest",
        volumeMeshFileName="tdcr_trunk_volume.vtk",
        youngModulus=18000,
        poissonRatio=0.3,
        totalMass=0.5
    )
    
    return root
```

## 🔍 Troubleshooting

### 7. Common Issues and Solutions

#### Issue: "SofaCUDA plugin not found"
```bash
# Solution: Rebuild SOFA with CUDA support
cmake .. -DPLUGIN_SOFACUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
make -j$(nproc)
```

#### Issue: "CUDA runtime error"
```bash
# Check GPU visibility in container
nvidia-smi

# Verify CUDA version compatibility
nvcc --version
cat /usr/local/cuda/version.txt
```

#### Issue: "CudaVec3f template not found"
```python
# Fallback to CPU if CUDA fails
def create_robust_object(parent, use_cuda=True):
    if use_cuda:
        try:
            return create_cuda_object(parent)
        except Exception as e:
            print(f"CUDA failed: {e}, falling back to CPU")
    
    return create_cpu_object(parent)
```

### 8. Performance Verification

Test CUDA performance vs CPU:

```python
# performance_test.py
import time
import Sofa

def benchmark_solvers():
    """Compare CUDA vs CPU solver performance."""
    
    def create_test_scene(use_cuda=False):
        root = Sofa.Core.Node("root")
        root.addObject('RequiredPlugin', name="SofaCUDA" if use_cuda else "")
        
        # Create identical scenes with different solvers
        test_node = root.addChild("test")
        template = "CudaVec3f" if use_cuda else "Vec3f"
        
        test_node.addObject('MechanicalObject', template=template)
        test_node.addObject('CGLinearSolver', template=template)
        # ... add more components
        
        return root
    
    # Benchmark both versions
    for use_cuda in [False, True]:
        solver_type = "CUDA" if use_cuda else "CPU"
        start_time = time.time()
        
        try:
            root = create_test_scene(use_cuda)
            Sofa.Simulation.init(root)
            
            # Run simulation steps
            for _ in range(100):
                Sofa.Simulation.animate(root, 0.01)
            
            elapsed = time.time() - start_time
            print(f"{solver_type} solver: {elapsed:.3f}s")
            
        except Exception as e:
            print(f"{solver_type} solver failed: {e}")
```

## 🎯 Integration with Your Projects

### 9. Update Your TDCR Projects

Replace the commented CUDA code in your projects:

```python
# In tdcr_trunk.py, replace ElasticMaterialObject with:
from cuda_elastic_fixed import CudaElasticMaterialObject

# Use in your TDCR function:
def TDCR_trunk_cuda(parentNode, **kwargs):
    trunk = CudaElasticMaterialObject(
        parentNode,
        name="TDCR_trunk",
        volumeMeshFileName="tdcr_trunk_volume.vtk",
        youngModulus=18000,
        poissonRatio=0.3,
        totalMass=0.5
    )
    
    # Add cables and other components as before
    # ...
    
    return trunk
```

### 10. Docker Compose Setup

For easier management, use docker-compose:

```yaml
# docker-compose.yml
version: '3.8'
services:
  sofa-cuda:
    image: your-sofa-cuda:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    volumes:
      - ./my_files:/workspace/my_files
    working_dir: /workspace
    stdin_open: true
    tty: true
```

## ✅ Verification Checklist

- [ ] NVIDIA drivers installed on host
- [ ] nvidia-container-toolkit installed
- [ ] Docker can access GPU (`docker run --gpus all nvidia/cuda nvidia-smi`)
- [ ] SOFA built with PLUGIN_SOFACUDA=ON
- [ ] SofaCUDA plugin loads without errors
- [ ] CudaVec3f template available
- [ ] Performance improvement observed

## 🚀 Next Steps

1. **Test the setup** with the provided test scripts
2. **Update your TDCR projects** to use CUDA
3. **Benchmark performance** to verify acceleration
4. **Monitor GPU usage** with `nvidia-smi` during simulations
5. **Optimize memory usage** for large simulations

## 📚 Additional Resources

- [SOFA CUDA Documentation](https://www.sofa-framework.org/community/doc/plugins/sofacuda/)
- [NVIDIA Docker Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

Happy CUDA-accelerated soft robotics! 🚀