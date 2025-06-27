#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUDA Setup Verification Script for SOFA
=======================================

This script tests whether CUDA is properly configured in your SOFA environment.
Run this inside your Docker container to verify CUDA acceleration is working.

Usage:
    python test_cuda_setup.py

Features:
- Tests CUDA plugin availability
- Verifies CUDA templates work
- Compares CPU vs CUDA performance
- Provides detailed diagnostics
"""

import sys
import time
import traceback
from typing import Optional, Dict, Any

try:
    import Sofa
    import Sofa.Core
    import Sofa.Simulation
    SOFA_AVAILABLE = True
except ImportError as e:
    print(f"❌ SOFA not available: {e}")
    SOFA_AVAILABLE = False


class CudaTestResults:
    """Container for test results."""

    def __init__(self):
        self.cuda_plugin_loaded = False
        self.cuda_templates_available = False
        self.cuda_solver_works = False
        self.cuda_mechanics_works = False
        self.cuda_fem_works = False
        self.performance_gain = 0.0
        self.errors = []
        self.warnings = []

    def is_cuda_working(self) -> bool:
        """Check if CUDA is fully functional."""
        return (self.cuda_plugin_loaded and
                self.cuda_templates_available and
                self.cuda_solver_works and
                self.cuda_mechanics_works)


def print_header():
    """Print test header."""
    print("🔬 SOFA CUDA Setup Verification")
    print("=" * 50)
    print()


def test_sofa_availability() -> bool:
    """Test if SOFA is available."""
    print("📦 Testing SOFA availability...")

    if not SOFA_AVAILABLE:
        print("❌ SOFA not available - check your SOFA installation")
        return False

    try:
        # Test basic SOFA functionality
        root = Sofa.Core.Node("test")
        print(f"✅ SOFA available - version info:")

        # Try to get version info
        if hasattr(Sofa, '__version__'):
            print(f"   Version: {Sofa.__version__}")
        else:
            print("   Version: Unknown")

        # Test simulation functions
        Sofa.Simulation.init(root)
        print("✅ SOFA simulation functions working")

        return True

    except Exception as e:
        print(f"❌ SOFA basic test failed: {e}")
        return False


def test_cuda_plugin(root_node: Sofa.Core.Node) -> bool:
    """Test if SofaCUDA plugin can be loaded."""
    print("\n🔌 Testing CUDA plugin loading...")

    try:
        # Try to load SofaCUDA plugin
        root_node.addObject('RequiredPlugin', name="SofaCUDA")
        print("✅ SofaCUDA plugin loaded successfully")

        # Initialize to make sure plugin actually loads
        Sofa.Simulation.init(root_node)
        print("✅ SofaCUDA plugin initialized successfully")

        return True

    except Exception as e:
        print(f"❌ Failed to load SofaCUDA plugin: {e}")
        print("   Make sure SOFA was compiled with PLUGIN_SOFACUDA=ON")
        return False


def test_cuda_templates(root_node: Sofa.Core.Node) -> bool:
    """Test if CUDA templates are available."""
    print("\n🧩 Testing CUDA templates...")

    test_node = root_node.addChild("cuda_template_test")

    cuda_templates = [
        "CudaVec3f",
        "CudaVec3d",
        "CudaRigid3f"
    ]

    working_templates = []

    for template in cuda_templates:
        try:
            # Try to create MechanicalObject with CUDA template
            mo = test_node.addObject('MechanicalObject',
                                   template=template,
                                   name=f'test_{template}')
            working_templates.append(template)
            print(f"✅ {template} template available")

        except Exception as e:
            print(f"❌ {template} template failed: {e}")

    if working_templates:
        print(f"✅ {len(working_templates)}/{len(cuda_templates)} CUDA templates working")
        return True
    else:
        print("❌ No CUDA templates available")
        return False


def test_cuda_solver(root_node: Sofa.Core.Node) -> bool:
    """Test CUDA linear solver."""
    print("\n⚙️  Testing CUDA solver...")

    test_node = root_node.addChild("cuda_solver_test")

    try:
        # Create simple scene with CUDA solver
        test_node.addObject('EulerImplicitSolver', name='odesolver')

        cuda_solver = test_node.addObject('CGLinearSolver',
                                        template='CudaVec3f',
                                        name='cuda_solver',
                                        iterations=25,
                                        tolerance=1e-6)

        test_node.addObject('MechanicalObject',
                          template='CudaVec3f',
                          name='dofs',
                          position=[[0,0,0], [1,0,0], [0,1,0], [0,0,1]])

        # Initialize to test if solver actually works
        Sofa.Simulation.init(root_node)

        print("✅ CUDA CGLinearSolver created and initialized")
        return True

    except Exception as e:
        print(f"❌ CUDA solver test failed: {e}")
        traceback.print_exc()
        return False


def test_cuda_mechanics(root_node: Sofa.Core.Node) -> bool:
    """Test CUDA mechanical components."""
    print("\n🔧 Testing CUDA mechanical components...")

    test_node = root_node.addChild("cuda_mechanics_test")

    try:
        # Create mechanical scene with CUDA
        test_node.addObject('EulerImplicitSolver')
        test_node.addObject('CGLinearSolver', template='CudaVec3f')

        # Create some test geometry
        positions = [[0,0,0], [1,0,0], [0,1,0], [0,0,1], [0.5,0.5,0.5]]

        mo = test_node.addObject('MechanicalObject',
                               template='CudaVec3f',
                               name='dofs',
                               position=positions)

        mass = test_node.addObject('UniformMass',
                                 totalMass=1.0)

        # Test constraint correction
        cc = test_node.addObject('UncoupledConstraintCorrection')

        # Initialize
        Sofa.Simulation.init(root_node)

        print("✅ CUDA mechanical components working")
        return True

    except Exception as e:
        print(f"❌ CUDA mechanics test failed: {e}")
        return False


def test_cuda_fem(root_node: Sofa.Core.Node) -> bool:
    """Test CUDA FEM force field."""
    print("\n🏗️  Testing CUDA FEM...")

    test_node = root_node.addChild("cuda_fem_test")

    try:
        # Create minimal tetrahedral mesh for testing
        test_node.addObject('EulerImplicitSolver')
        test_node.addObject('CGLinearSolver', template='CudaVec3f')

        # Simple tetrahedron
        positions = [[0,0,0], [1,0,0], [0,1,0], [0,0,1]]
        tetrahedra = [[0,1,2,3]]

        test_node.addObject('MechanicalObject',
                          template='CudaVec3f',
                          name='dofs',
                          position=positions)

        test_node.addObject('TetrahedronSetTopologyContainer',
                          name='container',
                          tetrahedra=tetrahedra)

        test_node.addObject('UniformMass', totalMass=1.0)

        # CUDA FEM force field
        fem = test_node.addObject('TetrahedronFEMForceField',
                                template='CudaVec3f',
                                name='FEM',
                                method='large',
                                poissonRatio=0.3,
                                youngModulus=1000)

        # Initialize
        Sofa.Simulation.init(root_node)

        print("✅ CUDA FEM force field working")
        return True

    except Exception as e:
        print(f"❌ CUDA FEM test failed: {e}")
        return False


def benchmark_performance(root_node: Sofa.Core.Node) -> float:
    """Benchmark CPU vs CUDA performance."""
    print("\n⚡ Performance benchmarking...")

    def create_test_scene(parent: Sofa.Core.Node, use_cuda: bool, name: str):
        """Create identical test scene with CPU or CUDA."""
        node = parent.addChild(name)

        template = "CudaVec3f" if use_cuda else "Vec3f"

        node.addObject('EulerImplicitSolver')

        if use_cuda:
            node.addObject('CGLinearSolver', template=template, iterations=10)
        else:
            node.addObject('CGLinearSolver', iterations=10)

        # Create larger test mesh
        import numpy as np
        n = 10  # Grid size
        positions = []
        tetrahedra = []

        # Generate simple grid of points
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    positions.append([i*0.1, j*0.1, k*0.1])

        # Add some tetrahedra (simplified)
        for i in range(min(len(positions)-4, 100)):
            tetrahedra.append([i, i+1, i+2, i+3])

        node.addObject('MechanicalObject',
                      template=template,
                      position=positions)

        node.addObject('TetrahedronSetTopologyContainer',
                      tetrahedra=tetrahedra)

        node.addObject('UniformMass', totalMass=10.0)

        node.addObject('TetrahedronFEMForceField',
                      template=template,
                      method='large',
                      poissonRatio=0.3,
                      youngModulus=5000)

        return node

    try:
        # Create CPU scene
        cpu_node = create_test_scene(root_node, False, "cpu_benchmark")

        # Create CUDA scene
        cuda_node = create_test_scene(root_node, True, "cuda_benchmark")

        # Initialize
        Sofa.Simulation.init(root_node)

        # Benchmark CPU
        print("   Benchmarking CPU...")
        start_time = time.time()
        for _ in range(50):
            Sofa.Simulation.animate(cpu_node, 0.01)
        cpu_time = time.time() - start_time

        # Benchmark CUDA
        print("   Benchmarking CUDA...")
        start_time = time.time()
        for _ in range(50):
            Sofa.Simulation.animate(cuda_node, 0.01)
        cuda_time = time.time() - start_time

        # Calculate speedup
        if cuda_time > 0:
            speedup = cpu_time / cuda_time
            print(f"   CPU time: {cpu_time:.3f}s")
            print(f"   CUDA time: {cuda_time:.3f}s")
            print(f"   Speedup: {speedup:.2f}x")
            return speedup
        else:
            print("   ⚠️  CUDA time measurement failed")
            return 0.0

    except Exception as e:
        print(f"   ❌ Performance benchmark failed: {e}")
        return 0.0


def run_cuda_verification() -> CudaTestResults:
    """Run complete CUDA verification suite."""

    if not SOFA_AVAILABLE:
        print("Cannot run tests without SOFA")
        return CudaTestResults()

    results = CudaTestResults()

    try:
        # Create root node for testing
        root = Sofa.Core.Node("cuda_test_root")

        # Run tests
        if test_cuda_plugin(root):
            results.cuda_plugin_loaded = True

            if test_cuda_templates(root):
                results.cuda_templates_available = True

                if test_cuda_solver(root):
                    results.cuda_solver_works = True

                    if test_cuda_mechanics(root):
                        results.cuda_mechanics_works = True

                        if test_cuda_fem(root):
                            results.cuda_fem_works = True

                            # Only benchmark if basic functionality works
                            results.performance_gain = benchmark_performance(root)

    except Exception as e:
        results.errors.append(f"Test suite failed: {e}")
        traceback.print_exc()

    return results


def print_final_report(results: CudaTestResults):
    """Print final test report."""
    print("\n" + "="*50)
    print("📊 FINAL REPORT")
    print("="*50)

    # Test results
    tests = [
        ("CUDA Plugin Loading", results.cuda_plugin_loaded),
        ("CUDA Templates", results.cuda_templates_available),
        ("CUDA Solver", results.cuda_solver_works),
        ("CUDA Mechanics", results.cuda_mechanics_works),
        ("CUDA FEM", results.cuda_fem_works)
    ]

    for test_name, passed in tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20} {status}")

    print()

    # Overall status
    if results.is_cuda_working():
        print("🎉 CUDA IS FULLY FUNCTIONAL!")
        print("   Your SOFA installation supports CUDA acceleration")

        if results.performance_gain > 1.0:
            print(f"   Performance gain: {results.performance_gain:.2f}x faster")

        print("\n✨ Next steps:")
        print("   1. Use cuda_elastic_working.py in your projects")
        print("   2. Replace ElasticMaterialObject with CudaElasticMaterialObject")
        print("   3. Monitor GPU usage with: nvidia-smi")

    else:
        print("💥 CUDA SETUP INCOMPLETE")
        print("   Some CUDA components are not working properly")

        print("\n🔧 Troubleshooting suggestions:")

        if not results.cuda_plugin_loaded:
            print("   • Rebuild SOFA with: cmake -DPLUGIN_SOFACUDA=ON")
            print("   • Ensure CUDA toolkit is installed in container")

        if not results.cuda_templates_available:
            print("   • Check CUDA runtime compatibility")
            print("   • Verify GPU is accessible: nvidia-smi")

        print("   • Check Docker GPU access: docker run --gpus all")
        print("   • Review cuda_setup_guide.md for detailed instructions")

    # Errors and warnings
    if results.errors:
        print(f"\n❌ Errors ({len(results.errors)}):")
        for error in results.errors:
            print(f"   • {error}")

    if results.warnings:
        print(f"\n⚠️  Warnings ({len(results.warnings)}):")
        for warning in results.warnings:
            print(f"   • {warning}")


def check_system_info():
    """Print system information for debugging."""
    print("\n🖥️  System Information")
    print("-" * 30)

    try:
        import platform
        print(f"Python: {platform.python_version()}")
        print(f"Platform: {platform.platform()}")

        # Check for NVIDIA GPU
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                gpu_names = result.stdout.strip().split('\n')
                print(f"GPU(s): {', '.join(gpu_names)}")
            else:
                print("GPU: No NVIDIA GPU detected")
        except:
            print("GPU: Cannot check (nvidia-smi not available)")

        # Check CUDA version
        try:
            result = subprocess.run(['nvcc', '--version'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Extract CUDA version
                for line in result.stdout.split('\n'):
                    if 'release' in line:
                        print(f"CUDA: {line.strip()}")
                        break
            else:
                print("CUDA: nvcc not found")
        except:
            print("CUDA: Cannot check version")

    except Exception as e:
        print(f"System info error: {e}")


def main():
    """Main test function."""
    print_header()

    # Basic checks
    if not test_sofa_availability():
        print("\n💥 Cannot proceed without SOFA")
        sys.exit(1)

    check_system_info()

    # Run CUDA tests
    print("\n🚀 Running CUDA verification tests...")
    results = run_cuda_verification()

    # Print results
    print_final_report(results)

    # Exit with appropriate code
    sys.exit(0 if results.is_cuda_working() else 1)


if __name__ == "__main__":
    main()
