# -*- coding: utf-8 -*-
import Sofa.Core
import Sofa.constants.Key as Key
from stlib3.physics.deformable import ElasticMaterialObject
from stlib3.physics.constraints import FixedBox
from softrobots.actuators import PullingCable
from stlib3.physics.collision import CollisionMesh
from splib3.loaders import loadPointListFromFile
from stlib3.scene import MainHeader, ContactHeader
import sys
import os
import csv
import numpy as np
from collections import defaultdict
import time
import threading
# from matplotlib import pyplot as plt
from cuda_elastic import CudaElasticMaterialObject
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
                      drawBoxes=False,
                      doUpdate=True,
                      strict=False)
        roi_nodes.append(roi)
    return roi_nodes

def add_monitors_to_rois(roi_nodes):
    """Add a Monitor to each ROI node for trajectory display only."""
    for idx, roi in enumerate(roi_nodes):
        roi.addObject("Monitor",
                      name="roiMonitor",
                      template="Vec3d",
                      listening=True,
                      indices=f"@roi_{idx+1}.indices",
                      showPositions=False,
                      PositionsColor=[1.0, 0.0, 0.0, 1.0],
                      showMinThreshold=0.01,
                      showTrajectories=False,
                      TrajectoriesPrecision=0.1,
                      TrajectoriesColor=[1,1,0,1],
                      ExportPositions=False)

def rotate_cable_points(points, deg, center=(19.75,0,19.75)):
       """Rotate a list of [x, y, z] points by deg degrees around the Y axis about center."""
       if deg == 0:
           return [list(pt) for pt in points]
       theta = np.deg2rad(deg)
       c, s = np.cos(theta), np.sin(theta)
       cx, cy, cz = center
       rotated = []
       for x, y, z in points:
           x0, z0 = x - cx, z - cz
           x1 = x0 * c - z0 * s
           z1 = x0 * s + z0 * c
           rotated.append([x1 + cx, y, z1 + cz])
       return rotated

def log_roi_csv(csv_file, cables, roi_nodes, soft_body_node, printInTerminal=1):
    dofs = soft_body_node.getObject('dofs')
    if dofs is None:
        if printInTerminal:
            print("MechanicalObject 'dofs' not found!")
        return
    positions = dofs.position.value
    avg_roi_coords = []
    for idx, roi in enumerate(roi_nodes):
        boxroi = roi.getObject(f"roi_{idx+1}")
        indices = boxroi.indices.value
        roi_coords = [positions[i] for i in indices]
        if roi_coords:
            avg_x = sum(pt[0] for pt in roi_coords) / len(roi_coords)
            avg_y = sum(pt[1] for pt in roi_coords) / len(roi_coords)
            avg_z = sum(pt[2] for pt in roi_coords) / len(roi_coords)
            avg_roi_coords += [f"{avg_x:.3f}", f"{avg_y:.3f}", f"{avg_z:.3f}"]
        else:
            avg_roi_coords += ["", "", ""]
    disp_values = [c.CableConstraint.value[0] for c in cables]
    force_values = [c.CableConstraint.force.value for c in cables]
    file_exists = os.path.isfile(csv_file)
    if (not file_exists or os.stat(csv_file).st_size == 0) and len(avg_roi_coords) > 0:
        header = ["L1", "L2", "L3", "F1", "F2", "F3"]
        n_boxes = len(avg_roi_coords) // 3
        for i in range(1, n_boxes + 1):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [f"{d:.3f}" for d in disp_values] +
            [f"{f:.3f}" for f in force_values] +
            avg_roi_coords
        )

def spine_log_roi_csv(cables, roi_nodes, soft_body_node, roi_box_centers, spine_csv_file, y_tol=1e-5, printInTerminal=1):
    dofs = soft_body_node.getObject('dofs')
    if dofs is None:
        if printInTerminal:
            print("MechanicalObject 'dofs' not found!")
        return
    positions = dofs.position.value
    y_groups = defaultdict(list)
    for idx, center in enumerate(roi_box_centers):
        y_val = center[1]
        found = False
        for y_key in y_groups:
            if abs(y_val - y_key) < y_tol:
                y_groups[y_key].append(idx)
                found = True
                break
        if not found:
            y_groups[y_val].append(idx)
    group_points = []
    for y_key in sorted(y_groups.keys()):
        indices_in_group = y_groups[y_key]
        points = []
        for idx in indices_in_group:
            roi = roi_nodes[idx]
            boxroi = roi.getObject(f"roi_{idx+1}")
            indices = boxroi.indices.value
            roi_coords = [positions[i] for i in indices]
            points.extend(roi_coords)
        if points:
            avg_x = sum(pt[0] for pt in points) / len(points)
            avg_y = sum(pt[1] for pt in points) / len(points)
            avg_z = sum(pt[2] for pt in points) / len(points)
            group_points.append((avg_x, avg_y, avg_z))
        else:
            group_points.append(("", "", ""))
    disp_values = [c.CableConstraint.value[0] for c in cables]
    force_values = [c.CableConstraint.force.value for c in cables]
    row = [f"{d:.3f}" for d in disp_values] + [f"{f:.3f}" for f in force_values]
    for pt in group_points:
        row += [f"{pt[0]:.3f}" if pt[0] != "" else "",
                f"{pt[1]:.3f}" if pt[1] != "" else "",
                f"{pt[2]:.3f}" if pt[2] != "" else ""]
    file_exists = os.path.isfile(spine_csv_file)
    if (not file_exists or os.stat(spine_csv_file).st_size == 0) and len(group_points) > 0:
        header = ["L1", "L2", "L3", "F1", "F2", "F3"]
        for i in range(1, len(group_points)+1):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        with open(spine_csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
    with open(spine_csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

class TDCR_trunk_Controller(Sofa.Core.Controller):
    def __init__(self, cable_nodes, roi_nodes, soft_body_node, roi_box_centers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cables = cable_nodes  # List of 3 cable nodes
        self.name = "TDCR_trunk_Controller"
        self.displacement_step = 0.1
        self.max_displacement = 10000
        self.min_displacement = -10000
        self.roi_nodes = roi_nodes
        self.roi_box_centers = roi_box_centers
        self.soft_body_node = soft_body_node
        self.roi_points = None

        self.csv_file = os.path.join(
            "/home/ci/my_files/TDCR_TRUNK/CSV_Plots", 
            "tdcr_trunk_output.csv"
        )
        self.spine_csv_file = os.path.join(
            "/home/ci/my_files/TDCR_TRUNK/CSV_Plots", 
            "tdcr_trunk_spine.csv"
        )
        # Key mappings for pull and release
        self.pull_keys = {"1": 0, "2": 1, "3": 2}
        self.release_keys = {"!": 0, "@": 1, "#": 2}

        # Initialize output directory and CSV file
        output_dir = "/home/ci/my_files/TDCR_TRUNK/CSV_Plots"
        csv_files = [
            os.path.join(output_dir, "tdcr_trunk_output.csv"),
            os.path.join(output_dir, "tdcr_trunk_spine.csv")
            ]
        for file in csv_files:
            with open(file, 'w'):
                pass 

    def cable_stepper(self, step_sizes, interval, steps):
        """
        Moves each cable by its step_size for a number of steps, with a pause between steps.
        Args:
            step_sizes: list/tuple of step sizes for each cable (e.g. [0.5, 0.5, 0.5])
            interval: seconds between steps
            steps: number of steps
        """
        def step_loop():
            for i in range(steps):
                for idx, step in enumerate(step_sizes):
                    self._adjust_cable(idx, step)
                print(f"Step {i+1}/{steps}: " +
                      ", ".join(f"L{j+1}={self.cables[j].CableConstraint.value[0]:.3f}" for j in range(len(self.cables))))
                log_roi_csv(self.csv_file, self.cables, self.roi_nodes, self.soft_body_node, printInTerminal=0)
                spine_log_roi_csv(self.cables, self.roi_nodes, self.soft_body_node, self.roi_box_centers, self.spine_csv_file, printInTerminal=0)
                time.sleep(interval)
            print("Cable stepping finished.")
        threading.Thread(target=step_loop, daemon=True).start()

    def cable_stepper_to_goal(self, step_sizes, interval, goals):
        """
        Moves each cable by its step_size every interval until it reaches its goal.
        Args:
            step_sizes: list of step sizes for each cable (e.g. [0.5, 0.5, 0.5])
            interval: seconds between steps
            goals: list of final destination values for each cable (e.g. [10.0, 10.0, 10.0])
        """
        def step_loop():
            while True:
                done = True
                for idx, (step, goal) in enumerate(zip(step_sizes, goals)):
                    current = self.cables[idx].CableConstraint.value[0]
                    diff = goal - current
                    if abs(diff) > abs(step):
                        move = step if diff > 0 else -abs(step)
                        self._adjust_cable(idx, move)
                        done = False
                    elif abs(diff) > 1e-6:
                        self._adjust_cable(idx, diff)
                # print("Cable positions: " + ", ".join(f"L{j+1}={self.cables[j].CableConstraint.value[0]:.3f}" for j in range(len(self.cables))))
                log_roi_csv(self.csv_file, self.cables, self.roi_nodes, self.soft_body_node, printInTerminal=0)
                spine_log_roi_csv(self.cables, self.roi_nodes, self.soft_body_node, self.roi_box_centers, self.spine_csv_file, printInTerminal=0)
                disp_values = [c.CableConstraint.value[0] for c in self.cables]
                print("Cable displacements: [{}]".format(", ".join(f"{d:.2f}" for d in disp_values)))
                force_values = [c.CableConstraint.force.value for c in self.cables]
                print("Applied forces: [{}]".format(", ".join(f"{f:.2f}" for f in force_values)))
                if done:
                    print("Cable stepping to goal finished.")
                    break
                time.sleep(interval)
        threading.Thread(target=step_loop, daemon=True).start()

    def onKeypressedEvent(self, event):
        
        key = event["key"]

        if key in self.pull_keys:
            idx = self.pull_keys[key]
            self._adjust_cable(idx, self.displacement_step)
        elif key in self.release_keys:
            idx = self.release_keys[key]
            self._adjust_cable(idx, -self.displacement_step)
            # self._adjust_cable(1, self.displacement_step)
#        # Constrained displacement of cables
        # if key == "1":
        #     if (self.cables[0].CableConstraint.value[0] + self.displacement_step <= self.max_displacement and
        #         self.cables[1].CableConstraint.value[0] - self.displacement_step >= self.min_displacement):
        #         self._adjust_cable(0, self.displacement_step)
        #         self._adjust_cable(1, -self.displacement_step)

        # elif key == "2":
        #     if (self.cables[0].CableConstraint.value[0] - self.displacement_step >= self.min_displacement and
        #         self.cables[1].CableConstraint.value[0] + self.displacement_step <= self.max_displacement):
        #         self._adjust_cable(0, -self.displacement_step)
        #         self._adjust_cable(1, self.displacement_step)


        # Contract all cables
        elif key == "4":
            for idx in range(len(self.cables)):
                self._adjust_cable(idx, self.displacement_step)
        elif key == "$":
            for idx in range(len(self.cables)):
                self._adjust_cable(idx, -self.displacement_step)
        # Automated movement: step to goal for all cables
        elif key == "0":
            # Example: move all cables to 10.0 in steps of 0.5, interval 0.2s
            self.cable_stepper_to_goal(step_sizes=[0.1, 0,0],interval= 0.5,goals= [200.0, 0,0])
            # self.cable_stepper_to_goal(step_sizes=[0.1, 0.1,0.1],interval= 0.1,goals= [20.0, 20.0, 20.0])

        disp_values = [c.CableConstraint.value[0] for c in self.cables]
        print("Cable displacements: [{}]".format(", ".join(f"{d:.2f}" for d in disp_values)))
        force_values = [c.CableConstraint.force.value for c in self.cables]
        print("Applied forces: [{}]".format(", ".join(f"{f:.2f}" for f in force_values)))

        log_roi_csv(self.csv_file, self.cables, self.roi_nodes, self.soft_body_node, printInTerminal=0)
        spine_log_roi_csv(self.cables, self.roi_nodes, self.soft_body_node, self.roi_box_centers, self.spine_csv_file, printInTerminal=0)

    def _adjust_cable(self, idx, delta):
        current_disp = self.cables[idx].CableConstraint.value[0]
        new_disp = current_disp + delta
        self.cables[idx].CableConstraint.value = [new_disp]

# fixingBox=[-1,0,-1,51.52,7,51.52]


x = 19.75
y = 272.68
z = 19.75
d = 1  # half the size of the fixing box
# 19.75 - 5.4 = 14.35
def TDCR_trunk(parentNode, name="TDCR_trunk",
         rotation=[0.0, 0.0, 0.0], translation=[0.0, 0.0, 0.0],
         fixingBox=[0,-4,0,39.5,10,39.50] , minForce = -sys.float_info.max, maxForce = sys.float_info.max):
# fixingBox=[-1,-4,-1,41.50,10,41.50] [x-d,y-d,z-d,x+d,y+d,z+d]
    tdcr = parentNode.addChild(name)
# 
    # Deformable object (visual + FEM)
    
    soft_body = ElasticMaterialObject(tdcr,
        volumeMeshFileName="tdcr_trunk_volume.vtk",
        surfaceMeshFileName="tdcr_trunk_surface.stl",
        collisionMesh="tdcr_trunk_collision.stl",
        withConstraint=False,
        youngModulus=8_000.0,  # Young's modulus in Pascals
        poissonRatio=0.20,
        totalMass=0.115,
        surfaceColor=[0.96, 0.87, 0.70, 1.0],
        rotation=rotation,
        translation=translation
    )

    # soft_body = CudaElasticMaterialObject(tdcr,
    #     volumeMeshFileName="tdcr_trunk_volume.vtk",
    #     surfaceMeshFileName="tdcr_trunk_surface.stl",
    #     collisionMesh="tdcr_trunk_collision.stl",
    #     withConstrain=False,
    #     youngModulus=8_000.0,  # Young's modulus in Pascals
    #     poissonRatio=0.20,
    #     totalMass=0.115,
    #     surfaceColor=[0.96, 0.87, 0.70, 1.0],
    #     rotation=rotation,
    #     translation=translation
    # )

    
    soft_body.collisionmodel.TriangleCollisionModel.selfCollision = True   
    soft_body.collisionmodel.LineCollisionModel.selfCollision = True
    soft_body.collisionmodel.PointCollisionModel.selfCollision = True

        # self.collisionmodel.createObject('TriangleCollisionModel')
        # self.collisionmodel.createObject('LineCollisionModel')
        # self.collisionmodel.createObject('PointCollisionModel')
    tdcr.addChild(soft_body)

    soft_body.addObject('LinearSolverConstraintCorrection')

    # soft_body.addObject('UncoupledConstraintCorrection')

    # soft_body.addObject('PrecomputedConstraintCorrection')

    # soft_body.addObject('GenericConstraintCorrection')

    # Fix the base
    FixedBox(soft_body, atPositions=fixingBox, doVisualization=True)

    



    
    #Add the Cables
    c1 = loadPointListFromFile("cable1.json")
    c1_r = rotate_cable_points(c1, 0)
    # Lower the pull point by 3 units in y direction

    pull1 = [c1_r[0][0], c1_r[0][1] - 3, c1_r[0][2]]
    cable1 = PullingCable(soft_body,
                          "PullingCable_1",
                          pullPointLocation=pull1,
                          rotation=rotation,
                          translation=translation,
                          cableGeometry=c1_r)
    cable1.CableConstraint.minForce = minForce
    cable1.CableConstraint.maxForce = maxForce

    delta = 0
    # For cable 2: rotate and shift all y by +delta, then remove last point
    c2_r = rotate_cable_points(c1_r, 120)
    c2_r = [[x, y + delta, z] for x, y, z in c2_r]
    # c2_r = c2_r[:-2]  # Remove last point
    pull2 = [c2_r[0][0], c2_r[0][1] - 3, c2_r[0][2]]
    cable2 = PullingCable(soft_body,
                          "PullingCable_2",
                          pullPointLocation=pull2,
                          rotation=rotation,
                          translation=translation,
                          cableGeometry=c2_r)
    cable2.CableConstraint.minForce = minForce
    cable2.CableConstraint.maxForce = maxForce

    # For cable 3: rotate and shift all y by +delta, then remove last point
    c3_r = rotate_cable_points(c1_r, 240)
    c3_r = [[x, y + delta, z] for x, y, z in c3_r]
    # c3_r = c3_r[:-2]  # Remove last point
    pull3 = [c3_r[0][0], c3_r[0][1] - 3, c3_r[0][2]]
    cable3 = PullingCable(soft_body,
                          "PullingCable_3",
                          pullPointLocation=pull3,
                          rotation=rotation,
                          translation=translation,
                          cableGeometry=c3_r)
    cable3.CableConstraint.minForce = minForce
    cable3.CableConstraint.maxForce = maxForce
    
    roi_centers = c1_r + c2_r + c3_r
    epsilons = [5.0] * len(roi_centers)
    roi_boxes = make_roi_boxes(roi_centers, epsilons)
    roi_nodes = add_boxrois(soft_body, roi_boxes)
    add_monitors_to_rois(roi_nodes)

    # controller = TDCR_trunk_Controller([cable1,cable2],roi_node=roi, soft_body_node=soft_body)
    controller = TDCR_trunk_Controller([cable1, cable2, cable3], roi_nodes=roi_nodes, soft_body_node=soft_body, roi_box_centers=roi_centers)
    soft_body.addObject(controller)

    tdcr.addObject('EulerImplicitSolver', rayleighStiffness=0.5, rayleighMass=0.5)


    # Collision model

    # CollisionMesh(soft_body,
    #     name="TDCR_trunk_Collision",
    #     surfaceMeshFileName="tdcr_trunk_collision.stl",
    #     rotation=rotation,
    #     translation=translation,
    #     collisionGroup=1,
    #     selfCollision=True
    # )




    return tdcr


def loadRequiredPlugins(rootNode):
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.AnimationLoop') # Needed to use components [FreeMotionAnimationLoop]  
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Collision.Detection.Algorithm') # Needed to use components [BVHNarrowPhase,BruteForceBroadPhase,CollisionPipeline]  
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Collision.Detection.Intersection') # Needed to use components [LocalMinDistance]  
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Collision.Geometry') # Needed to use components [LineCollisionModel,PointCollisionModel,TriangleCollisionModel]  
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Collision.Response.Contact') # Needed to use components [RuleBasedContactManager]  
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Constraint.Lagrangian.Correction') # Needed to use components [LinearSolverConstraintCorrection]  
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Constraint.Lagrangian.Solver') # Needed to use components [GenericConstraintSolver]  
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Engine.Select') # Needed to use components [BoxROI]  
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.LinearSolver.Direct') # Needed to use components [SparseLDLSolver]  
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Mapping.Linear') # Needed to use components [BarycentricMapping]  
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Mass') # Needed to use components [UniformMass]  
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.ODESolver.Backward') # Needed to use components [EulerImplicitSolver]  
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.SolidMechanics.FEM.Elastic') # Needed to use components [TetrahedronFEMForceField]  
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.SolidMechanics.Spring') # Needed to use components [RestShapeSpringsForceField]  
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.StateContainer') # Needed to use components [MechanicalObject]  
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Topology.Container.Constant') # Needed to use components [MeshTopology]  
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Topology.Container.Dynamic') # Needed to use components [TetrahedronSetTopologyContainer]  
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Visual') # Needed to use components [VisualStyle] 
    rootNode.addObject('CollisionPipeline', name='collisionPipeline')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.MechanicalLoad') # Needed to use components [ConstantForceField]
    rootNode.addObject('RequiredPlugin', name='SofaCUDA')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.IO.Mesh') # Needed to use components [MeshVTKLoader]  
    rootNode.addObject('RequiredPlugin', name='Sofa.GL.Component.Rendering3D') # Needed to use components [OglSceneFrame]
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.LinearSolver.Iterative')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Constraint.Lagrangian.Correction')
    rootNode.addObject('RequiredPlugin', name='SofaValidation') # Needed to use components [Monitor]
    return

def createScene(rootNode):
    
    loadRequiredPlugins(rootNode)
    MainHeader(rootNode, gravity=[0.0, 0.0, 0.0], 
               plugins=["SoftRobots"])

    ContactHeader(rootNode, 
                  alarmDistance=2.0, 
                  contactDistance=0.5, 
                  frictionCoef=0.02)
    rootNode.bbox = "-50 -50 -50 50 50 50"
    # rootNode.VisualStyle.displayFlags = "showVisual showInteractionForceFields showWireframe"
    rootNode.VisualStyle.displayFlags = "showVisual showInteractionForceFields"


    TDCR_trunk(rootNode,minForce=0.1)
    # TDCR_trunk(rootNode)


    return rootNode
'''tracking: position from standard tdcr code
validation: NelderMead optimization
theta optimization
images'''