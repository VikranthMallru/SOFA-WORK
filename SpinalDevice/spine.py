# -*- coding: utf-8 -*-

import Sofa.Core
import Sofa.Simulation
from stlib3.physics.deformable import ElasticMaterialObject
from stlib3.physics.constraints import FixedBox
from softrobots.actuators import PullingCable
from stlib3.physics.collision import CollisionMesh
from splib3.loaders import loadPointListFromFile
from stlib3.scene import MainHeader, ContactHeader
from stlib3.physics.rigid import RigidObject
import sys
import os
import math

def make_roi_boxes(coords, epsilons):
    """Create box definitions for each point and epsilon."""
    return [[x-e, y-e, z-e, x+e, y+e, z+e] for (x, y, z), e in zip(coords, epsilons)]

def add_boxrois(parent_node, roi_boxes):
    """Add BoxROI objects for each box definition."""
    roi_nodes = []
    for idx, box in enumerate(roi_boxes):
        roi = parent_node.addChild(f"SpinalROI_{idx+1}")
        roi.addObject('BoxROI',
                     name=f"spinal_roi_{idx+1}",
                     template="Vec3d",
                     box=box,
                     drawBoxes=False,
                     doUpdate=True,
                     strict=False)
        roi_nodes.append(roi)
    return roi_nodes

def add_monitors_to_rois(roi_nodes):
    """Add a Monitor to each ROI node for spinal tracking."""
    for idx, roi in enumerate(roi_nodes):
        roi.addObject("Monitor",
                     name="spinalRoiMonitor",
                     template="Vec3d",
                     listening=True,
                     indices=f"@spinal_roi_{idx+1}.indices",
                     showPositions=True,
                     PositionsColor=[0.0, 1.0, 0.0, 1.0],  # Green for spinal monitoring
                     showMinThreshold=0.01,
                     showTrajectories=False,
                     TrajectoriesPrecision=0.1,
                     TrajectoriesColor=[0,1,1,1],
                     ExportPositions=True)

class SpinalAssistiveController(Sofa.Core.Controller):
    def __init__(self, 
                 tendon_nodes,
                 roi_nodes,
                 soft_body_node,
                 root=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "SpinalAssistiveController"
        self.displacement_step = 0.3  # Smaller steps for spinal correction
        
        # Key mappings for 6 groups of 2 tendons each (12 total)
        # Group 1: Centre Full (tendons 0, 1)
        # Group 2: Centre Half (tendons 2, 3)
        # Group 3: Right Full (tendons 4, 5)
        # Group 4: Right Half (tendons 6, 7)
        # Group 5: Left Full (tendons 8, 9)
        # Group 6: Left Half (tendons 10, 11)
        
        self.tendon_groups = {
            "1": [0, 1],    # Centre Full
            "2": [2, 3],    # Centre Half
            "3": [4, 5],    # Right Full
            "4": [6, 7],    # Right Half
            "5": [8, 9],    # Left Full
            "6": [10, 11]   # Left Half
        }
        
        self.release_groups = {
            "!": [0, 1],    # Release Centre Full
            "@": [2, 3],    # Release Centre Half
            "#": [4, 5],    # Release Right Full
            "$": [6, 7],    # Release Right Half
            "%": [8, 9],    # Release Left Full
            "^": [10, 11]   # Release Left Half
        }
        
        self.group_names = {
            "1": "Centre Full",
            "2": "Centre Half", 
            "3": "Right Full",
            "4": "Right Half",
            "5": "Left Full",
            "6": "Left Half"
        }
        
        self.tendons = tendon_nodes
        self.roi_nodes = roi_nodes
        self.soft_body_node = soft_body_node
        self.root = root
        self.listening = True
    
    def onKeypressedEvent(self, event):
        key = event['key']
        print(f"Spinal Control Key Pressed: {key}")
        
        # Group activation
        if key in self.tendon_groups:
            tendon_indices = self.tendon_groups[key]
            group_name = self.group_names[key]
            for idx in tendon_indices:
                self._adjust_tendon(idx, self.displacement_step)
            print(f"Activating {group_name} (Tendons {tendon_indices[0]+1} & {tendon_indices[1]+1})")
            
        # Group release
        elif key in self.release_groups:
            tendon_indices = self.release_groups[key]
            for idx in tendon_indices:
                self._adjust_tendon(idx, -self.displacement_step)
            print(f"Releasing tendon group (Tendons {tendon_indices[0]+1} & {tendon_indices[1]+1})")
            
        # All tendons control
        elif key == "7":
            # Pull all tendons
            for idx in range(12):
                self._adjust_tendon(idx, self.displacement_step)
            print("Activating all 12 tendons")
            
        elif key == "&":
            # Release all tendons
            for idx in range(12):
                self._adjust_tendon(idx, -self.displacement_step)
            print("Releasing all 12 tendons")
        
        # Status display
        disp_values = [t.CableConstraint.value[0] for t in self.tendons]
        print("Tendon displacements: [{}]".format(
            ", ".join(f"T{i+1}:{d:.2f}" for i, d in enumerate(disp_values))))
        
        force_values = [t.CableConstraint.force.value for t in self.tendons]
        print("Applied forces: [{}]".format(
            ", ".join(f"T{i+1}:{f:.2f}" for i, f in enumerate(force_values))))
    
    def _adjust_tendon(self, idx, delta):
        current_disp = self.tendons[idx].CableConstraint.value[0]
        new_disp = max(0, current_disp + delta)
        self.tendons[idx].CableConstraint.value = [new_disp]
# [50,-160,80,-50,-180,-50]
# Utility function to compute box coordinates from a center and epsilon
def compute_box_from_center(center, epsilon):
    x, y, z = center
    return [x - epsilon, y - epsilon, z - epsilon, x + epsilon, y + epsilon, z + epsilon]
# [40,-180,15]
# [40,-180,-15]
# [-40,-180,15]
# [-40,-180,-15]
def SpinalAssistiveDevice(parentNode, 
                         name="SpinalAssistive",
                         rotation=[0.0, 0.0, 0.0], 
                         translation=[0.0, 0.0, 0.0],
                        #  fixingBox=compute_box_from_center([-40,-180,-15],1),  # Adjust for your spinal device
                        fixingBox=[50, -160, 80, -50, -200, -50],  # Adjust for your spinal device
                         minForce=-sys.float_info.max, 
                         maxForce=sys.float_info.max,
                         volume_mesh="spinal_volume.vtk",      # Your spinal device mesh
                         surface_mesh="spinal_surface.stl",    # Your spinal device surface
                         collision_mesh="spinal_collision.stl"): # Your spinal device collision

    # Create main spinal device node
    spinal_device = parentNode.addChild(name)
    
    # Add soft body using your STL file
    soft_body = ElasticMaterialObject(
        spinal_device,
        volumeMeshFileName=volume_mesh,
        surfaceMeshFileName=surface_mesh,
        collisionMesh=collision_mesh,
        youngModulus=1000,  # Adjusted for spinal tissue properties
        poissonRatio=0.35,   # More suitable for soft tissue
        totalMass=0.08,      # Adjusted for spinal device
        surfaceColor=[0.8, 0.6, 0.9, 1.0],  # Purple for spinal device
        rotation=rotation,
        translation=translation,
        withConstraint=False
    )
    
    spinal_device.addChild(soft_body)
    
    # Fix the base (lower spine attachment)
    FixedBox(soft_body, atPositions=fixingBox, doVisualization=True)
    
    # Define JSON file names based on group names
    tendon_files = [
        "centre_full_1.json",   # Group 1: Centre Full - Tendon 1
        "centre_full_2.json",   # Group 1: Centre Full - Tendon 2
        "centre_half_1.json",   # Group 2: Centre Half - Tendon 1
        "centre_half_2.json",   # Group 2: Centre Half - Tendon 2
        "right_full_1.json",    # Group 3: Right Full - Tendon 1
        "right_full_2.json",    # Group 3: Right Full - Tendon 2
        "right_half_1.json",    # Group 4: Right Half - Tendon 1
        "right_half_2.json",    # Group 4: Right Half - Tendon 2
        "left_full_1.json",     # Group 5: Left Full - Tendon 1
        "left_full_2.json",     # Group 5: Left Full - Tendon 2
        "left_half_1.json",     # Group 6: Left Half - Tendon 1
        "left_half_2.json"      # Group 6: Left Half - Tendon 2
    ]
    
    # Load tendon paths from JSON files
    tendon_paths = []
    for filename in tendon_files:
        tendon_path = loadPointListFromFile(filename)
        tendon_paths.append(tendon_path)
        print(f"Loaded {filename}")
    
    # Use all loaded tendon points for ROI coordinates
    all_tendon_points = []
    for path in tendon_paths:
        all_tendon_points.extend(path)
    
    # Use all tendon points as ROI coordinates
    spinal_roi_coords = all_tendon_points
    
    epsilon = 4  # Adjust epsilon for your device
    epsilons = [epsilon] * len(spinal_roi_coords)
    
    roi_boxes = make_roi_boxes(spinal_roi_coords, epsilons)
    roi_nodes = add_boxrois(soft_body, roi_boxes)
    add_monitors_to_rois(roi_nodes)
    
    # Create 12 tendons using the loaded paths
    tendons = []
    tendon_display_names = [
        "CentreFull_1", "CentreFull_2",      # Group 1: Centre Full
        "CentreHalf_1", "CentreHalf_2",      # Group 2: Centre Half
        "RightFull_1", "RightFull_2",        # Group 3: Right Full
        "RightHalf_1", "RightHalf_2",        # Group 4: Right Half
        "LeftFull_1", "LeftFull_2",          # Group 5: Left Full
        "LeftHalf_1", "LeftHalf_2"           # Group 6: Left Half
    ]
    
    for i, (name, path) in enumerate(zip(tendon_display_names, tendon_paths)):
        pull_point = [path[0][0], path[0][1] - 5, path[0][2]]
        
        tendon = PullingCable(
            soft_body,
            name,
            pullPointLocation=pull_point,
            rotation=rotation,
            translation=translation,
            cableGeometry=path
        )
        tendon.CableConstraint.minForce = minForce
        tendon.CableConstraint.maxForce = maxForce
        tendons.append(tendon)
        soft_body.addChild(tendon)
    
    # Add controller
    controller = SpinalAssistiveController(
        tendon_nodes=tendons,
        roi_nodes=roi_nodes,
        soft_body_node=soft_body,
        root=parentNode
    )
    
    soft_body.addObject(controller)
    
    # Enable self-collision for realistic spinal behavior
    soft_body.collisionmodel.TriangleCollisionModel.selfCollision = True
    soft_body.collisionmodel.LineCollisionModel.selfCollision = True
    soft_body.collisionmodel.PointCollisionModel.selfCollision = True
    
    spinal_device.addObject('EulerImplicitSolver', rayleighStiffness=0.05, rayleighMass=0.05)
    
    return spinal_device

def loadRequiredPlugins(rootNode):
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.AnimationLoop')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Collision.Detection.Algorithm')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Collision.Detection.Intersection')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Collision.Geometry')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Collision.Response.Contact')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Constraint.Lagrangian.Correction')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Constraint.Lagrangian.Solver')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Engine.Select')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.LinearSolver.Direct')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Mapping.Linear')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Mass')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.ODESolver.Backward')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.SolidMechanics.FEM.Elastic')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.SolidMechanics.Spring')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.StateContainer')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Topology.Container.Constant')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Topology.Container.Dynamic')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Visual')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.MechanicalLoad')
    rootNode.addObject('RequiredPlugin', name='SofaValidation')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.LinearSolver.Iterative')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Mapping.NonLinear')
    rootNode.addObject('CollisionPipeline', name='collisionPipeline')

def createScene(rootNode):
    loadRequiredPlugins(rootNode)
    
    MainHeader(rootNode, 
               gravity=[0.0, -981.0, 0.0],
               plugins=["SoftRobots"])
    
    ContactHeader(rootNode,
                  alarmDistance=1.5,
                  contactDistance=0.2,
                  frictionCoef=0.12)
    
    rootNode.bbox = "-30 -30 -30 30 150 30"
    rootNode.VisualStyle.displayFlags = "showVisual showInteractionForceFields"
    
    # Create the spinal assistive device
    SpinalAssistiveDevice(rootNode,
                         volume_mesh="spinal_volume.vtk",     # Replace with your file
                         surface_mesh="spinal_surface.stl",   # Replace with your file
                         collision_mesh="spinal_collision.stl", # Replace with your file
                         minForce=0.05)
    
    return rootNode
