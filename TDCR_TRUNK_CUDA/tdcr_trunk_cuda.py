# -*- coding: utf-8 -*-
"""
CUDA-Accelerated TDCR Trunk Simulation
======================================

A high-performance tendon-driven continuum robot trunk simulation using CUDA
acceleration for real-time soft robotics applications.

This implementation combines:
- CUDA-accelerated elastic material objects
- Multi-cable tendon actuation system
- Real-time ROI monitoring and data export
- Interactive control system
- Performance optimization for large deformations

Features:
- GPU acceleration with automatic CPU fallback
- Real-time simulation with cable-driven actuation
- CSV data export for analysis
- Interactive keyboard control
- ROI-based position monitoring
- Collision detection and response
- Visual feedback and monitoring

Author: AI Assistant
Date: 2024
License: MIT
"""

import Sofa
import Sofa.Core
import Sofa.constants.Key as Key
from cuda_elastic import CudaElasticMaterialObject
from softrobots.actuators import PullingCable
from splib3.loaders import loadPointListFromFile
from stlib3.scene import MainHeader, ContactHeader
from stlib3.physics.constraints import FixedBox
import sys
import os
import csv
import time
from collections import defaultdict
import numpy as np


def make_roi_boxes(coords, epsilons):
    """Create box definitions for each point and epsilon."""
    return [[x-e, y-e, z-e, x+e, y+e, z+e] for (x, y, z), e in zip(coords, epsilons)]


def add_boxrois(parent_node, roi_boxes):
    """Add BoxROI objects for each box definition."""
    roi_nodes = []
    for idx, box in enumerate(roi_boxes):
        roi = parent_node.addChild(f"ROI_{idx+1}")
        roi.addObject('BoxROI',
                      name=f"roi_{idx+1}",
                      template="Vec3d",
                      box=box,
                      drawBoxes=True,
                      doUpdate=True,
                      strict=False)
        roi_nodes.append(roi)
    return roi_nodes


def add_monitors_to_rois(roi_nodes):
    """Add a Monitor to each ROI node."""
    for idx, roi in enumerate(roi_nodes):
        roi.addObject("Monitor",
            name="roiMonitor",
            template="Vec3d",
            listening=True,
            indices=f"@roi_{idx+1}.indices",
            showPositions=True,
            PositionsColor=[1.0, 0.0, 0.0, 1.0],
            showMinThreshold=0.01,
            showTrajectories=True,
            TrajectoriesPrecision=0.1,
            TrajectoriesColor=[1,1,0,1],
            ExportPositions=False)


class TDCR_Trunk_CUDA_Controller(Sofa.Core.Controller):
    """
    Advanced controller for CUDA-accelerated TDCR trunk simulation.

    Provides:
    - Interactive cable control
    - Real-time data logging
    - Performance monitoring
    - ROI position tracking
    - CUDA performance statistics
    """

    def __init__(self, cable_nodes, roi_nodes, soft_body_node, cuda_object, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cables = cable_nodes  # List of cable nodes
        self.roi_nodes = roi_nodes
        self.soft_body_node = soft_body_node
        self.cuda_object = cuda_object
        self.name = "TDCR_Trunk_CUDA_Controller"

        # Control parameters
        self.displacement_step = 0.1
        self.max_displacement = 21.9
        self.min_displacement = -21.9

        # Key mappings for cable control
        self.pull_keys = {"1": 0, "2": 1}  # Keys 1,2 pull cables
        self.release_keys = {"!": 0, "@": 1}  # Keys Shift+1,2 release cables

        # Data logging setup
        self.output_dir = os.path.join(os.path.dirname(__file__), "CSV_Plots")
        os.makedirs(self.output_dir, exist_ok=True)

        self.csv_file = os.path.join(self.output_dir, "tdcr_trunk_cuda_output.csv")
        self.performance_file = os.path.join(self.output_dir, "cuda_performance.csv")

        # Initialize CSV files
        self._init_csv_files()

        # Performance monitoring
        self.step_count = 0
        self.total_time = 0.0
        self.cuda_active = self.cuda_object.is_cuda_enabled()

        # Status display
        self._print_initial_status()

    def _init_csv_files(self):
        """Initialize CSV files with headers."""
        # Main simulation data
        with open(self.csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            headers = ["timestamp", "step", "cable1_displacement", "cable2_displacement"]

            # Add ROI position headers
            for i, roi in enumerate(self.roi_nodes):
                headers.extend([f"roi_{i+1}_x", f"roi_{i+1}_y", f"roi_{i+1}_z"])

            # Add CUDA performance headers
            headers.extend(["cuda_enabled", "step_time_ms", "fps"])
            writer.writerow(headers)

        # Performance data
        with open(self.performance_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "step", "acceleration", "step_time_ms",
                           "fps", "young_modulus", "total_mass"])

    def _print_initial_status(self):
        """Print initial controller status."""
        print(f"\n🎮 TDCR Trunk CUDA Controller Initialized")
        print(f"   • Acceleration: {'CUDA ⚡' if self.cuda_active else 'CPU'}")
        print(f"   • Template: {self.cuda_object.get_template()}")
        print(f"   • Cables: {len(self.cables)}")
        print(f"   • ROIs: {len(self.roi_nodes)}")
        print(f"   • Output: {os.path.basename(self.csv_file)}")
        print(f"\n🎯 Controls:")
        print(f"   • Press '1' or '2' to pull cables")
        print(f"   • Press 'Shift+1' or 'Shift+2' to release cables")
        print(f"   • Press 'Space' to log current state")
        print(f"   • Press 'P' for performance report")
        print(f"   • Press 'R' to reset simulation")

    def onKeypressedEvent(self, event):
        """Handle keyboard input for cable control and data logging."""
        key = event["key"]
        start_time = time.time()

        # Cable control
        if key in self.pull_keys:
            idx = self.pull_keys[key]
            self._adjust_cable(idx, self.displacement_step)
            print(f"🔧 Pulling cable {idx+1}")

        elif key in self.release_keys:
            idx = self.release_keys[key]
            self._adjust_cable(idx, -self.displacement_step)
            print(f"🔧 Releasing cable {idx+1}")

        # Data logging and analysis
        elif key == Key.space:
            self._log_current_state()
            print("📊 State logged to CSV")

        elif key == "P" or key == "p":
            self._print_performance_report()

        elif key == "R" or key == "r":
            self._reset_simulation()
            print("🔄 Simulation reset")

        elif key == "H" or key == "h":
            self._print_help()

        # Update performance metrics
        step_time = (time.time() - start_time) * 1000  # ms
        self._update_performance_metrics(step_time)

    def _adjust_cable(self, cable_idx, displacement_change):
        """Adjust cable displacement with bounds checking."""
        if cable_idx >= len(self.cables):
            print(f"⚠️  Cable {cable_idx+1} not found")
            return

        cable = self.cables[cable_idx]
        current_displacement = cable.CableConstraint.value[0]
        new_displacement = current_displacement + displacement_change

        # Apply bounds
        new_displacement = max(self.min_displacement,
                             min(self.max_displacement, new_displacement))

        cable.CableConstraint.value = [new_displacement]

        # Update display
        if hasattr(self, 'last_cable_update'):
            if time.time() - self.last_cable_update > 0.5:  # Update every 0.5s
                print(f"Cable {cable_idx+1}: {new_displacement:.1f}")
                self.last_cable_update = time.time()
        else:
            self.last_cable_update = time.time()

    def _log_current_state(self):
        """Log current simulation state to CSV."""
        timestamp = time.time()

        # Get cable displacements
        cable_displacements = []
        for cable in self.cables:
            cable_displacements.append(cable.CableConstraint.value[0])

        # Get ROI positions
        roi_positions = self._get_roi_positions()

        # Prepare data row
        row = [timestamp, self.step_count] + cable_displacements

        # Add ROI positions (flattened)
        for pos in roi_positions:
            row.extend(pos)

        # Add performance data
        step_time = getattr(self, 'last_step_time', 0.0)
        fps = 1000.0 / step_time if step_time > 0 else 0.0

        row.extend([self.cuda_active, step_time, fps])

        # Write to CSV
        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def _get_roi_positions(self):
        """Get current positions of all ROI centers."""
        positions = []

        for roi_node in self.roi_nodes:
            try:
                # Get the monitor component
                monitor = None
                for obj in roi_node.objects:
                    if hasattr(obj, 'name') and 'Monitor' in obj.name:
                        monitor = obj
                        break

                if monitor and hasattr(monitor, 'positions'):
                    # Get average position of monitored points
                    pos_data = monitor.positions
                    if len(pos_data) > 0:
                        avg_pos = np.mean(pos_data, axis=0)
                        positions.append(avg_pos.tolist())
                    else:
                        positions.append([0.0, 0.0, 0.0])
                else:
                    positions.append([0.0, 0.0, 0.0])

            except Exception as e:
                print(f"⚠️  Error getting ROI position: {e}")
                positions.append([0.0, 0.0, 0.0])

        return positions

    def _update_performance_metrics(self, step_time):
        """Update performance tracking metrics."""
        self.step_count += 1
        self.last_step_time = step_time
        self.total_time += step_time

        # Log performance data periodically
        if self.step_count % 100 == 0:
            self._log_performance_data()

    def _log_performance_data(self):
        """Log performance data to CSV."""
        timestamp = time.time()
        avg_step_time = self.total_time / self.step_count if self.step_count > 0 else 0
        avg_fps = 1000.0 / avg_step_time if avg_step_time > 0 else 0

        # Get material properties
        young_modulus = self.cuda_object.young_modulus
        total_mass = self.cuda_object.total_mass

        row = [
            timestamp, self.step_count,
            "CUDA" if self.cuda_active else "CPU",
            avg_step_time, avg_fps,
            young_modulus, total_mass
        ]

        with open(self.performance_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def _print_performance_report(self):
        """Print current performance statistics."""
        if self.step_count == 0:
            print("📊 No performance data available yet")
            return

        avg_step_time = self.total_time / self.step_count
        avg_fps = 1000.0 / avg_step_time if avg_step_time > 0 else 0

        print(f"\n📊 Performance Report (Step {self.step_count})")
        print(f"   • Acceleration: {'CUDA ⚡' if self.cuda_active else 'CPU'}")
        print(f"   • Template: {self.cuda_object.get_template()}")
        print(f"   • Avg step time: {avg_step_time:.2f}ms")
        print(f"   • Avg FPS: {avg_fps:.1f}")
        print(f"   • Total time: {self.total_time/1000:.1f}s")
        print(f"   • Young's modulus: {self.cuda_object.young_modulus:,} Pa")

        # Cable status
        print(f"   • Cable displacements:")
        for i, cable in enumerate(self.cables):
            disp = cable.CableConstraint.value[0]
            print(f"     Cable {i+1}: {disp:.1f}")

    def _reset_simulation(self):
        """Reset simulation to initial state."""
        # Reset cable displacements
        for cable in self.cables:
            cable.CableConstraint.value = [0.0]

        # Reset performance counters
        self.step_count = 0
        self.total_time = 0.0

    def _print_help(self):
        """Print help information."""
        print(f"\n📖 TDCR Trunk CUDA Controller Help")
        print(f"================================")
        print(f"Cable Control:")
        print(f"  '1', '2'         - Pull cables 1, 2")
        print(f"  'Shift+1', 'Shift+2' - Release cables 1, 2")
        print(f"\nData & Analysis:")
        print(f"  'Space'          - Log current state to CSV")
        print(f"  'P'              - Show performance report")
        print(f"  'R'              - Reset simulation")
        print(f"  'H'              - Show this help")
        print(f"\nFiles:")
        print(f"  Data: {os.path.basename(self.csv_file)}")
        print(f"  Performance: {os.path.basename(self.performance_file)}")


def TDCR_trunk_cuda(parentNode=None,
                   name="TDCR_trunk_cuda",
                   rotation=[0.0, 0.0, 0.0],
                   translation=[0.0, 0.0, 0.0],
                   fixingBox=[-15.0, -15.0, -40.0, 15.0, 15.0, -35.0],
                   pullPointLocation=[0.0, 0.0, 0.0],
                   enable_cuda=True,
                   verbose=True):
    """
    Create CUDA-accelerated TDCR trunk with advanced monitoring capabilities.

    Args:
        parentNode: Parent SOFA node
        name: Object name
        rotation: Rotation angles [rx, ry, rz]
        translation: Translation [tx, ty, tz]
        fixingBox: Fixed constraint box [xmin, ymin, zmin, xmax, ymax, zmax]
        pullPointLocation: Cable pull point location
        enable_cuda: Enable CUDA acceleration
        verbose: Enable verbose output

    Returns:
        SOFA node containing the complete TDCR trunk system
    """

    if verbose:
        print(f"\n🚀 Creating CUDA-enabled TDCR Trunk")
        print(f"   • CUDA: {'Enabled' if enable_cuda else 'Disabled'}")

    # Create main node
    trunk = parentNode.addChild(name)

    # Create CUDA-accelerated elastic object
    try:
        trunk_body = CudaElasticMaterialObject(
            parent_node=trunk,
            name="TrunkBody",
            volume_mesh_filename="tdcr_trunk_volume.vtk",
            surface_mesh_filename="tdcr_trunk_surface.stl",
            collision_mesh_filename="tdcr_trunk_collision.stl",
            young_modulus=18000,
            poisson_ratio=0.3,
            total_mass=0.5,
            rotation=rotation,
            translation=translation,
            surface_color=[0.8, 0.3, 0.3, 1.0],
            solver_iterations=50,
            solver_tolerance=1e-6,
            enable_visual=True,
            enable_collision=True,
            verbose=verbose
        )

        if verbose:
            print(f"   • Elastic body: {'CUDA' if trunk_body.is_cuda_enabled() else 'CPU'}")

    except Exception as e:
        print(f"❌ Failed to create CUDA elastic object: {e}")
        print("   Falling back to standard implementation...")

        # Fallback to standard elastic object
        from stlib3.physics.deformable import ElasticMaterialObject
        trunk_body = ElasticMaterialObject(
            trunk,
            volumeMeshFileName="tdcr_trunk_volume.vtk",
            poissonRatio=0.3,
            youngModulus=18000,
            totalMass=0.5,
            surfaceColor=[0.8, 0.3, 0.3, 1.0],
            surfaceMeshFileName="tdcr_trunk_surface.stl",
            rotation=rotation,
            translation=translation
        )

        # Create a wrapper to maintain interface compatibility
        class FallbackWrapper:
            def __init__(self, elastic_obj):
                self.node = elastic_obj
                self.young_modulus = 18000
                self.total_mass = 0.5

            def is_cuda_enabled(self):
                return False

            def get_template(self):
                return "Vec3f"

            def get_node(self):
                return self.node

        trunk_body = FallbackWrapper(trunk_body)

    # Add fixed constraints
    FixedBox(trunk_body.get_node(),
            atPositions=fixingBox,
            doVisualization=True)

    # Create cable actuation system
    cables = []
    cable_files = ["cable1.json", "cable2.json"]

    for i, cable_file in enumerate(cable_files):
        if os.path.exists(cable_file):
            try:
                cable = PullingCable(
                    trunk_body.get_node(),
                    f"PullingCable_{i+1}",
                    pullPointLocation=pullPointLocation,
                    rotation=rotation,
                    translation=translation,
                    cableGeometry=loadPointListFromFile(cable_file)
                )
                cables.append(cable)

                if verbose:
                    print(f"   • Cable {i+1}: {cable_file}")

            except Exception as e:
                print(f"⚠️  Failed to load cable {i+1}: {e}")
        else:
            print(f"⚠️  Cable file not found: {cable_file}")

    # ROI monitoring setup
    roi_coords = [
        [0, 0, -10],   # Tip region
        [0, 0, -20],   # Middle region
        [0, 0, -30]    # Base region
    ]
    roi_epsilons = [2.0, 2.0, 2.0]

    # Create ROI boxes and monitors
    roi_boxes = make_roi_boxes(roi_coords, roi_epsilons)
    roi_nodes = add_boxrois(trunk_body.get_node(), roi_boxes)
    add_monitors_to_rois(roi_nodes)

    if verbose:
        print(f"   • ROI monitors: {len(roi_nodes)}")

    # Create controller
    controller = TDCR_Trunk_CUDA_Controller(
        cables, roi_nodes, trunk_body.get_node(), trunk_body
    )
    trunk.addObject(controller)

    if verbose:
        print(f"   • Controller: Advanced CUDA controller")
        print(f"✅ TDCR Trunk CUDA setup complete")

    return trunk


def createScene(rootNode):
    """
    Create the complete CUDA-accelerated TDCR trunk simulation scene.

    This function sets up:
    - CUDA-enabled physics simulation
    - Collision detection and response
    - Real-time cable actuation
    - Performance monitoring
    - Data export capabilities
    """

    print("🌟 CUDA-Accelerated TDCR Trunk Simulation")
    print("=" * 50)

    # Scene configuration
    MainHeader(rootNode,
              gravity=[0.0, -981.0, 0.0],
              plugins=["SoftRobots", "SofaCUDA"])

    ContactHeader(rootNode,
                 alarmDistance=4,
                 contactDistance=3,
                 frictionCoef=0.08)

    # Visual settings
    rootNode.VisualStyle.displayFlags = "showBehavior showCollisionModels"

    # Create CUDA-accelerated TDCR trunk
    trunk = TDCR_trunk_cuda(
        rootNode,
        name="TDCR_Trunk_CUDA",
        translation=[0.0, 0.0, 0.0],
        verbose=True
    )

    # Print final status
    print("\n🎮 Simulation ready!")
    print("   Press 'H' in simulation for controls help")
    print("   Data will be saved to CSV_Plots/ directory")

    return rootNode


def main():
    """Main function for standalone execution."""
    import Sofa.Gui

    # Create root node
    root = Sofa.Core.Node("root")

    # Create scene
    createScene(root)

    # Initialize simulation
    Sofa.Simulation.init(root)

    # Launch GUI
    Sofa.Gui.GUIManager.Init("TDCR_Trunk_CUDA", "qglviewer")
    Sofa.Gui.GUIManager.createGUI(root, __file__)
    Sofa.Gui.GUIManager.SetDimension(1200, 800)

    # Start simulation
    Sofa.Gui.GUIManager.MainLoop(root)
    Sofa.Gui.GUIManager.closeGUI()


if __name__ == '__main__':
    main()
