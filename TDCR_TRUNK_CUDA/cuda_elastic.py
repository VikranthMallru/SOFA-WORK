# -*- coding: utf-8 -*-
"""
CUDA Elastic Material Object for TDCR Trunk Simulation
======================================================

A robust CUDA-accelerated elastic material object specifically designed
for the TDCR (Tendon-Driven Continuum Robot) trunk simulation.

This implementation provides significant performance improvements through
GPU acceleration while maintaining compatibility with CPU fallback.

Features:
- Automatic CUDA/CPU detection and fallback
- Optimized for soft robotics simulations
- Compatible with SoftRobots plugin
- Real-time performance monitoring
- Robust error handling

Author: AI Assistant
Date: 2024
"""

import Sofa
import Sofa.Core
import os
from typing import Optional, List, Union


class CudaElasticMaterialObject:
    """
    CUDA-accelerated elastic material object for SOFA soft robotics simulations.

    This class creates a deformable continuum robot body with GPU acceleration
    for improved performance in real-time simulations.
    """

    def __init__(self,
                 parent_node: Sofa.Core.Node,
                 name: str = "CudaElastic",
                 volume_mesh_filename: Optional[str] = None,
                 surface_mesh_filename: Optional[str] = None,
                 collision_mesh_filename: Optional[str] = None,
                 young_modulus: float = 18000.0,
                 poisson_ratio: float = 0.3,
                 total_mass: float = 1.0,
                 rotation: List[float] = None,
                 translation: List[float] = None,
                 scale: List[float] = None,
                 surface_color: List[float] = None,
                 solver_iterations: int = 50,
                 solver_tolerance: float = 1e-6,
                 force_field_method: str = "large",
                 enable_constraints: bool = True,
                 enable_visual: bool = True,
                 enable_collision: bool = True,
                 verbose: bool = False):
        """
        Initialize CUDA-accelerated elastic material object.

        Args:
            parent_node: Parent SOFA node
            name: Object name
            volume_mesh_filename: Path to volume mesh (.vtk)
            surface_mesh_filename: Path to surface mesh (.stl)
            collision_mesh_filename: Path to collision mesh (optional)
            young_modulus: Young's modulus (Pa)
            poisson_ratio: Poisson's ratio
            total_mass: Total mass (kg)
            rotation: Rotation [rx, ry, rz] degrees
            translation: Translation [tx, ty, tz]
            scale: Scale [sx, sy, sz]
            surface_color: RGBA color [r, g, b, a]
            solver_iterations: Linear solver iterations
            solver_tolerance: Solver tolerance
            force_field_method: FEM method
            enable_constraints: Enable constraint correction
            enable_visual: Enable visual model
            enable_collision: Enable collision detection
            verbose: Enable debug output
        """

        # Set defaults
        self.rotation = rotation or [0.0, 0.0, 0.0]
        self.translation = translation or [0.0, 0.0, 0.0]
        self.scale = scale or [1.0, 1.0, 1.0]
        self.surface_color = surface_color or [0.7, 0.3, 0.3, 1.0]

        # Store parameters
        self.volume_mesh_filename = volume_mesh_filename
        self.surface_mesh_filename = surface_mesh_filename
        self.collision_mesh_filename = collision_mesh_filename
        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio
        self.total_mass = total_mass
        self.solver_iterations = solver_iterations
        self.solver_tolerance = solver_tolerance
        self.force_field_method = force_field_method
        self.enable_constraints = enable_constraints
        self.enable_visual = enable_visual
        self.enable_collision = enable_collision
        self.verbose = verbose

        # Create main node
        self.node = parent_node.addChild(name)

        # CUDA state
        self.cuda_available = False
        self.template = "Vec3f"  # Default to CPU

        # Initialize
        self._check_cuda_availability()
        self._create_components()

        if self.verbose:
            self._print_summary()

    def _check_cuda_availability(self):
        """Check and configure CUDA availability."""
        try:
            root = self.node.getRoot()

            # Load SofaCUDA plugin
            if not self._has_plugin(root, "SofaCUDA"):
                try:
                    root.addObject('RequiredPlugin', name="SofaCUDA")
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  SofaCUDA plugin not available: {e}")
                    return

            # Test CUDA functionality
            test_node = root.addChild("_cuda_test_temp")
            try:
                test_mo = test_node.addObject('MechanicalObject',
                                            template='CudaVec3f',
                                            name='test_cuda_mo')

                test_solver = test_node.addObject('CGLinearSolver',
                                                template='CudaVec3f',
                                                name='test_cuda_solver')

                # If we reach here, CUDA is working
                self.cuda_available = True
                self.template = "CudaVec3f"

                if self.verbose:
                    print("✅ CUDA acceleration enabled")

            except Exception as e:
                if self.verbose:
                    print(f"⚠️  CUDA not available, using CPU: {e}")
                self.cuda_available = False
                self.template = "Vec3f"

            finally:
                # Clean up test node
                try:
                    root.removeChild(test_node)
                except:
                    pass

        except Exception as e:
            if self.verbose:
                print(f"⚠️  CUDA check failed: {e}")
            self.cuda_available = False
            self.template = "Vec3f"

    def _has_plugin(self, root: Sofa.Core.Node, plugin_name: str) -> bool:
        """Check if plugin is loaded."""
        for obj in root.objects:
            if hasattr(obj, 'name') and obj.name == plugin_name:
                return True
        return False

    def _create_components(self):
        """Create all SOFA components."""
        if not self.volume_mesh_filename:
            raise ValueError("Volume mesh filename is required")

        if not os.path.exists(self.volume_mesh_filename):
            raise FileNotFoundError(f"Volume mesh not found: {self.volume_mesh_filename}")

        self._load_volume_mesh()
        self._create_solvers()
        self._create_topology_and_mechanics()
        self._create_force_field()

        if self.enable_constraints:
            self._create_constraint_correction()

        if self.enable_visual and self.surface_mesh_filename:
            self._create_visual_model()

        if self.enable_collision:
            self._create_collision_model()

    def _load_volume_mesh(self):
        """Load volume mesh."""
        filename = self.volume_mesh_filename.lower()

        if filename.endswith('.vtk'):
            self.loader = self.node.addObject('MeshVTKLoader',
                                            name='loader',
                                            filename=self.volume_mesh_filename,
                                            rotation=self.rotation,
                                            translation=self.translation,
                                            scale3d=self.scale)
        elif filename.endswith('.msh'):
            self.loader = self.node.addObject('MeshGmshLoader',
                                            name='loader',
                                            filename=self.volume_mesh_filename,
                                            rotation=self.rotation,
                                            translation=self.translation,
                                            scale3d=self.scale)
        else:
            raise ValueError(f"Unsupported mesh format: {filename}")

    def _create_solvers(self):
        """Create solvers."""
        # ODE Solver
        self.ode_solver = self.node.addObject('EulerImplicitSolver',
                                            name='odesolver',
                                            rayleighStiffness=0.1,
                                            rayleighMass=0.1)

        # Linear Solver
        if self.cuda_available:
            self.linear_solver = self.node.addObject('CGLinearSolver',
                                                   template=self.template,
                                                   name='solver',
                                                   iterations=self.solver_iterations,
                                                   tolerance=self.solver_tolerance,
                                                   threshold=1e-9)
        else:
            # CPU fallback
            self.linear_solver = self.node.addObject('SparseLDLSolver',
                                                   name='solver')

    def _create_topology_and_mechanics(self):
        """Create topology and mechanical objects."""
        # Topology
        self.topology = self.node.addObject('TetrahedronSetTopologyContainer',
                                          src='@loader',
                                          name='container')

        # Mechanical Object
        self.mechanical_object = self.node.addObject('MechanicalObject',
                                                    template=self.template,
                                                    name='dofs',
                                                    src='@loader')

        # Mass
        self.mass = self.node.addObject('UniformMass',
                                      totalMass=self.total_mass,
                                      name='mass')

    def _create_force_field(self):
        """Create FEM force field."""
        self.force_field = self.node.addObject('TetrahedronFEMForceField',
                                             template=self.template,
                                             name='FEM',
                                             method=self.force_field_method,
                                             poissonRatio=self.poisson_ratio,
                                             youngModulus=self.young_modulus,
                                             updateStiffnessMatrix=False,
                                             computeGlobalMatrix=False)

    def _create_constraint_correction(self):
        """Create constraint correction."""
        if self.cuda_available:
            self.constraint_correction = self.node.addObject('UncoupledConstraintCorrection',
                                                           name='constraintCorrection')
        else:
            self.constraint_correction = self.node.addObject('LinearSolverConstraintCorrection',
                                                           solverName='solver')

    def _create_visual_model(self):
        """Create visual model."""
        if not self.surface_mesh_filename or not os.path.exists(self.surface_mesh_filename):
            if self.verbose:
                print(f"⚠️  Surface mesh not found, using volume mesh for visualization")
            self._create_volume_visual()
            return

        # Create visual node
        self.visual_node = self.node.addChild('VisualModel')

        # Load surface mesh
        self.visual_loader = self.visual_node.addObject('MeshSTLLoader',
                                                      name='loader',
                                                      filename=self.surface_mesh_filename,
                                                      rotation=self.rotation,
                                                      translation=self.translation,
                                                      scale3d=self.scale)

        # Visual model
        self.visual_model = self.visual_node.addObject('OglModel',
                                                     name='visualModel',
                                                     src='@loader',
                                                     color=self.surface_color,
                                                     updateNormals=True)

        # Mapping
        self.visual_mapping = self.visual_node.addObject('BarycentricMapping',
                                                        name='mapping',
                                                        input='@../dofs',
                                                        output='@visualModel')

    def _create_volume_visual(self):
        """Create visual model from volume mesh."""
        self.visual_node = self.node.addChild('VisualModel')

        self.visual_model = self.visual_node.addObject('OglModel',
                                                     name='visualModel',
                                                     src='@../loader',
                                                     color=self.surface_color,
                                                     updateNormals=True)

        self.visual_mapping = self.visual_node.addObject('BarycentricMapping',
                                                        name='mapping',
                                                        input='@../dofs',
                                                        output='@visualModel')

    def _create_collision_model(self):
        """Create collision model."""
        # Create collision node
        self.collision_node = self.node.addChild('CollisionModel')

        # Use collision mesh if available, otherwise surface mesh
        collision_file = self.collision_mesh_filename or self.surface_mesh_filename

        if collision_file and os.path.exists(collision_file):
            # Load collision mesh
            self.collision_loader = self.collision_node.addObject('MeshSTLLoader',
                                                                name='loader',
                                                                filename=collision_file,
                                                                rotation=self.rotation,
                                                                translation=self.translation,
                                                                scale3d=self.scale)

            self.collision_topology = self.collision_node.addObject('TriangleSetTopologyContainer',
                                                                   src='@loader',
                                                                   name='container')
        else:
            # Extract surface from volume mesh
            self.collision_topology = self.collision_node.addObject('TriangleSetTopologyContainer',
                                                                   name='container')
            self.collision_node.addObject('Tetra2TriangleTopologicalMapping',
                                         input='@../container',
                                         output='@container')

        # Collision mechanical object (CPU for collision detection)
        self.collision_dofs = self.collision_node.addObject('MechanicalObject',
                                                           template='Vec3d',
                                                           name='dofs')

        # Collision models
        self.triangle_collision = self.collision_node.addObject('TriangleCollisionModel',
                                                               name='triangleModel',
                                                               group=1)

        self.line_collision = self.collision_node.addObject('LineCollisionModel',
                                                           name='lineModel',
                                                           group=1)

        self.point_collision = self.collision_node.addObject('PointCollisionModel',
                                                            name='pointModel',
                                                            group=1)

        # Mapping from simulation to collision
        self.collision_mapping = self.collision_node.addObject('BarycentricMapping',
                                                              name='mapping',
                                                              input='@../dofs',
                                                              output='@dofs')

    def _print_summary(self):
        """Print object summary."""
        print(f"\n🤖 TDCR Trunk CUDA Object '{self.node.name}':")
        print(f"   • Acceleration: {'CUDA ⚡' if self.cuda_available else 'CPU'}")
        print(f"   • Template: {self.template}")
        print(f"   • Volume mesh: {os.path.basename(self.volume_mesh_filename)}")
        print(f"   • Young's modulus: {self.young_modulus:,.0f} Pa")
        print(f"   • Mass: {self.total_mass} kg")
        print(f"   • Visual: {'✅' if self.enable_visual else '❌'}")
        print(f"   • Collision: {'✅' if self.enable_collision else '❌'}")

    # Public interface methods
    def get_node(self) -> Sofa.Core.Node:
        """Get the SOFA node."""
        return self.node

    def get_mechanical_object(self):
        """Get mechanical object."""
        return self.mechanical_object

    def get_force_field(self):
        """Get FEM force field."""
        return self.force_field

    def is_cuda_enabled(self) -> bool:
        """Check if CUDA is enabled."""
        return self.cuda_available

    def get_template(self) -> str:
        """Get current template."""
        return self.template

    def set_young_modulus(self, young_modulus: float):
        """Update Young's modulus."""
        self.young_modulus = young_modulus
        if hasattr(self, 'force_field'):
            self.force_field.youngModulus = young_modulus

    def set_poisson_ratio(self, poisson_ratio: float):
        """Update Poisson's ratio."""
        self.poisson_ratio = poisson_ratio
        if hasattr(self, 'force_field'):
            self.force_field.poissonRatio = poisson_ratio


def create_cuda_test_scene():
    """Create test scene with CUDA elastic object."""

    def createScene(rootNode):
        # Basic scene setup
        rootNode.addObject('RequiredPlugin', name="SofaCUDA")
        rootNode.addObject('RequiredPlugin', name="SoftRobots")
        rootNode.addObject('VisualStyle', displayFlags="showBehavior showForceFields")

        # Animation loop
        rootNode.addObject('FreeMotionAnimationLoop')
        rootNode.addObject('GenericConstraintSolver', tolerance=1e-6, maxIterations=1000)

        # Collision pipeline
        rootNode.addObject('CollisionPipeline')
        rootNode.addObject('BruteForceBroadPhase')
        rootNode.addObject('BVHNarrowPhase')
        rootNode.addObject('LocalMinDistance', alarmDistance=5, contactDistance=2)
        rootNode.addObject('CollisionResponse', response="FrictionContactConstraint")

        # Create CUDA elastic object
        cuda_trunk = CudaElasticMaterialObject(
            parent_node=rootNode,
            name="CudaTrunk",
            volume_mesh_filename="tdcr_trunk_volume.vtk",
            surface_mesh_filename="tdcr_trunk_surface.stl",
            collision_mesh_filename="tdcr_trunk_collision.stl",
            young_modulus=18000,
            poisson_ratio=0.3,
            total_mass=0.5,
            surface_color=[0.8, 0.3, 0.3, 1.0],
            verbose=True
        )

        return rootNode

    return createScene


# Performance comparison utility
class PerformanceMonitor:
    """Monitor simulation performance for CUDA vs CPU comparison."""

    def __init__(self):
        self.cuda_times = []
        self.cpu_times = []

    def time_simulation(self, root_node, steps=100, dt=0.01):
        """Time simulation performance."""
        import time

        start_time = time.time()
        for _ in range(steps):
            Sofa.Simulation.animate(root_node, dt)
        elapsed = time.time() - start_time

        return elapsed

    def compare_performance(self, cuda_node, cpu_node, steps=50):
        """Compare CUDA vs CPU performance."""
        print("\n⚡ Performance Comparison")
        print("-" * 30)

        # Time CUDA
        cuda_time = self.time_simulation(cuda_node, steps)
        print(f"CUDA time: {cuda_time:.3f}s")

        # Time CPU
        cpu_time = self.time_simulation(cpu_node, steps)
        print(f"CPU time: {cpu_time:.3f}s")

        # Calculate speedup
        if cuda_time > 0:
            speedup = cpu_time / cuda_time
            print(f"Speedup: {speedup:.2f}x")

            if speedup > 1.5:
                print("🚀 Significant CUDA acceleration!")
            elif speedup > 1.0:
                print("✅ CUDA provides some acceleration")
            else:
                print("⚠️  CUDA may not be optimal for this scene size")

        return cuda_time, cpu_time


if __name__ == "__main__":
    print("CUDA Elastic Material Object for TDCR Trunk")
    print("=" * 50)
    print("This module provides CUDA-accelerated elastic objects")
    print("for improved soft robotics simulation performance.")
    print("\nUsage:")
    print("  from cuda_elastic import CudaElasticMaterialObject")
    print("  trunk = CudaElasticMaterialObject(parent_node, ...)")
