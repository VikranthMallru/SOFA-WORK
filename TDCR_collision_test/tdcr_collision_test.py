# -*- coding: utf-8 -*-
import Sofa.Core
# import Sofa.constants.Key as Key
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
import csv
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
                      drawBoxes=False,
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
            showPositions=False,
            PositionsColor=[1.0, 0.0, 0.0, 1.0],
            showMinThreshold=0.01,
            showTrajectories=True,
            TrajectoriesPrecision=0.1,
            TrajectoriesColor=[1,1,0,1],
            ExportPositions=False
        )

def add_forcefields_to_rois(roi_nodes, forces):
    """Add a ConstantForceField to each ROI node, with a different force."""
    for roi, force in zip(roi_nodes, forces):
        roi.addObject("ConstantForceField",
                      name="roiConstForce",
                      force=force)

def log_roi_csv(csv_file, cables, roi_nodes, soft_body_node, printInTerminal=1):
    """Log cable, force, and ROI average point data to CSV, print all points and averages in terminal if printInTerminal is set."""
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
        if printInTerminal:
            print(f"roi_{idx+1} contains {len(roi_coords)} points:")
        for pt_idx, (x, y, z) in enumerate(roi_coords):
            if printInTerminal:
                print(f"  Point {pt_idx+1}: ({x:.3f}, {y:.3f}, {z:.3f})")
        # Print and compute average for this ROI
        if roi_coords:
            avg_x = sum(pt[0] for pt in roi_coords) / len(roi_coords)
            avg_y = sum(pt[1] for pt in roi_coords) / len(roi_coords)
            avg_z = sum(pt[2] for pt in roi_coords) / len(roi_coords)
            if printInTerminal:
                print(f"  Average: ({avg_x:.3f}, {avg_y:.3f}, {avg_z:.3f})")
            avg_roi_coords += [f"{avg_x:.3f}", f"{avg_y:.3f}", f"{avg_z:.3f}"]
        else:
            if printInTerminal:
                print("  Average: (N/A, N/A, N/A)")
            avg_roi_coords += ["", "", ""]  # Handle empty ROI gracefully

    disp_values = [c.CableConstraint.value[0] for c in cables]
    force_values = [c.CableConstraint.force.value for c in cables]
    file_exists = os.path.isfile(csv_file)
    # Write header only if file doesn't exist or is empty
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

def add_rigid_object_from_stl(parent_node,
                              name="rigidObject",
                              stl_path="mesh/object.stl",
                              translation=[0.0, 0.0, 0.0],
                              rotation=[0.0, 0.0, 0.0],
                              scale=1.0,
                              total_mass=1.0,
                              volume=1.0,
                              inertia_matrix=None,
                              color=[0.8, 0.2, 0.2],
                              isStatic=False):
    """
    Adds a rigid object from an STL mesh to the parent_node, using SOFA native components.
    If isStatic is True, the object will be fixed (static) in the scene, like the floor.
    """
    show_object_scale = scale
    if inertia_matrix is None:
        inertia_matrix = [0., 0., 0., 0., 0., 0., 0., 0., 0.]

    rigid = parent_node.addChild(name)

    # Only add solvers and mass if not static
    if not isStatic:
        rigid.addObject('EulerImplicitSolver', name='odesolver')
        rigid.addObject('CGLinearSolver', name='Solver', iterations=25, tolerance=1e-05, threshold=1e-05)
        rigid.addObject('MechanicalObject', name="mstate", template="Rigid3",
                        translation2=translation, rotation2=rotation, showObjectScale=show_object_scale)
        rigid.addObject('UniformMass', name="mass", vertexMass=[total_mass, volume, inertia_matrix[:]])
        rigid.addObject('UncoupledConstraintCorrection')
    else:
        # For static, just add the MechanicalObject (no solver, no mass)
        rigid.addObject('MechanicalObject', name="mstate", template="Rigid3",
                        translation2=translation, rotation2=rotation, showObjectScale=show_object_scale)

    # Collision subnode using STL
    collision = rigid.addChild('collision')
    collision.addObject('MeshSTLLoader', name="loader", filename=stl_path, scale=scale)
    collision.addObject('MeshTopology', src="@loader")
    collision.addObject('MechanicalObject')

    # Set moving=False, simulated=False if static (like your floor)
    collision.addObject('TriangleCollisionModel', moving= not isStatic, simulated=not isStatic)
    collision.addObject('LineCollisionModel', moving=not isStatic, simulated=not isStatic)
    collision.addObject('PointCollisionModel', moving=not isStatic, simulated=not isStatic)

    collision.addObject('RigidMapping')

    # Visualization subnode using STL
    visu = rigid.addChild("VisualModel")
    visu.loader = visu.addObject('MeshSTLLoader', name="loader", filename=stl_path)
    visu.addObject('OglModel', name="model", src="@loader", scale3d=[show_object_scale]*3, color=color, updateNormals=False)
    visu.addObject('RigidMapping')

    return rigid

def rotate_cable_points(points, deg, center=(17.5, 0, 17.5)):
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


class TDCRController(Sofa.Core.Controller):
    def __init__(self,
                 cable_nodes,
                 roi_nodes,
                 soft_body_node,
                 csv_file,
                 root= None,
                 initial_theta_deg=0.0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "TDCRController"
        # self.displacement_step = 0.1
        self.displacement_step = 0.5
        self.pull_keys = {"1": 0, "2": 1, "3": 2}
        self.release_keys = {"!": 0, "@": 1, "#": 2}
        self.cables = cable_nodes
        self.roi_nodes = roi_nodes
        self.soft_body_node = soft_body_node
        self.csv_file = csv_file
        self.root = root
        self.initial_theta_deg = initial_theta_deg
        self.listening = True


        # Clean the CSV file at every restart
        os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)
        with open(self.csv_file, "w", newline="") as f:
            pass

    def onKeypressedEvent(self, event):

        key = event['key']
        print("pressed")
        # Existing cable control logic
        if key in self.pull_keys:
            idx = self.pull_keys[key]
            self._adjust_cable(idx, self.displacement_step)
        elif key in self.release_keys:
            idx = self.release_keys[key]
            self._adjust_cable(idx, -self.displacement_step)
            # if key == "!" and self.cables[0].CableConstraint.value[0] <= 0:
            #     # self._adjust_cable(0, -self.displacement_step)
            #     self._adjust_cable(1, self.displacement_step)
            #     self._adjust_cable(2, self.displacement_step)
                
        elif key == "4":
            for idx in range(3):
                self._adjust_cable(idx, self.displacement_step)
        elif key == "$":
            for idx in range(3):
                self._adjust_cable(idx, -self.displacement_step)


        # if key != "4" and key != "$":
        log_roi_csv(self.csv_file,
                    self.cables,
                    self.roi_nodes,
                    self.soft_body_node,
                    printInTerminal=0
                    )

        # Existing status display
        disp_values = [c.CableConstraint.value[0] for c in self.cables]
        print("Cable displacements: [{}]".format(", ".join(f"{d:.2f}" for d in disp_values)))
        force_values = [c.CableConstraint.force.value for c in self.cables]
        print("Applied forces: [{}]".format(", ".join(f"{f:.2f}" for f in force_values)))

    def _adjust_cable(self, idx, delta):
        current_disp = self.cables[idx].CableConstraint.value[0]
        new_disp = max(0, current_disp + delta)
        self.cables[idx].CableConstraint.value = [new_disp]

  
def TDCR(parentNode, name="TDCR",
         rotation=[0.0, 0.0, 0.0], translation=[0.0, 0.0, 0.0],
         fixingBox=[36, -1, -1, -1, 6, 36], minForce = -sys.float_info.max, maxForce = sys.float_info.max,
         initial_theta_deg=0.0):
    #############################################################################################################
   
    # adding soft body
    tdcr = parentNode.addChild(name)
    #############################################################################################################

    soft_body = ElasticMaterialObject(tdcr,
        volumeMeshFileName="tdcr_volume.vtk",
        surfaceMeshFileName="tdcr_surface.stl",
        collisionMesh="tdcr_collision.stl",
        youngModulus=60000,
        poissonRatio=0.25,
        totalMass=0.03,
        surfaceColor=[0.96, 0.87, 0.70, 1.0],
        rotation=rotation,
        translation=translation,
        withConstraint=False
    )
    tdcr.addChild(soft_body)

    soft_body.addObject('LinearSolverConstraintCorrection')
    # Fix the base
    FixedBox(soft_body, atPositions=fixingBox, doVisualization=True)
    #############################################################################################################
    # --- Define your ROI centers, epsilons, and forces here ---
    c1=rotate_cable_points(loadPointListFromFile("cable1.json"),initial_theta_deg)
    c2 = rotate_cable_points(c1,120)
    c3 = rotate_cable_points(c1,240)
    cables = []
    cable_points_list = [c1, c2, c3]
    for i, cable_points in enumerate(cable_points_list, start=1):
        cable = PullingCable(
            soft_body,
            f"PullingCable_{i}",
            pullPointLocation=cable_points[0],
            rotation=rotation,
            translation=translation,
            cableGeometry=cable_points
        )
        cables.append(cable)


    all_points = c1 + c2 + c3
    
    e=3
    epsilons = [e] * len(all_points)                            # epsilons
    forces = [[0,0,0], [0,0,0], [0,0,0]]           # forces, one per ROI

    roi_boxes = make_roi_boxes(all_points, epsilons)
    roi_nodes = add_boxrois(soft_body, roi_boxes)
    add_monitors_to_rois(roi_nodes)
    # add_forcefields_to_rois(roi_nodes, forces)


    #############################################################################################################
    # Instantiating the TDCRController


    output_dir = "/home/ci/my_files/TDCR_collision_test/CSV_Plots"
    csv_file = os.path.join(output_dir, "tdcr_output.csv")


    # def __init__(self,
    #              cable_nodes,
    #              roi_nodes,
    #              soft_body_node,
    #              csv_file,
    #              root= None,
    #              initial_theta_deg=0.0,
    #              *args, **kwargs):

    controller = TDCRController(
        cable_nodes=cables,
        roi_nodes=roi_nodes,
        soft_body_node=soft_body,
        csv_file=csv_file,
        root=parentNode,
        initial_theta_deg=initial_theta_deg
    )


    soft_body.collisionmodel.TriangleCollisionModel.selfCollision = True 
    soft_body.collisionmodel.LineCollisionModel.selfCollision = True
    soft_body.collisionmodel.PointCollisionModel.selfCollision = True

    soft_body.addObject(controller)


    tdcr.addObject('EulerImplicitSolver', rayleighStiffness=0.1, rayleighMass=0.1)
    #############################################################################################################
    # Loading the collision model
    # Collision model

    # CollisionMesh(soft_body,
    #     name="TDCRCollision",
    #     surfaceMeshFileName="tdcr_collision.stl",
    #     rotation=rotation,
    #     translation=translation,
    #     collisionGroup=1
    # )

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
    rootNode.addObject('RequiredPlugin', name='SofaValidation') # Needed to use components [Monitor]
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.LinearSolver.Iterative') # Needed to use components [CGLinearSolver]  
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Mapping.NonLinear') # Needed to use components [RigidMapping]
    return

def createScene(rootNode):

    loadRequiredPlugins(rootNode)
    MainHeader(rootNode, gravity=[0.0, -981.0, 0.0],
               plugins=["SoftRobots"])
    ContactHeader(rootNode,
                  alarmDistance=1.0,
                  contactDistance=0.1,
                  frictionCoef=0.08)
    rootNode.bbox = "-50 -50 -50 50 50 50"
    rootNode.VisualStyle.displayFlags = "showVisual showInteractionForceFields"
    TDCR(rootNode,
         initial_theta_deg=0.0,
         minForce=0.1)  # Set initial_theta_deg to 0.0 for no rotation

    # add_rigid_object_from_stl(
    # rootNode,  # or rootNode, or wherever you want it
    # name="RigidCube",
    # stl_path="cube.stl",
    # translation=[15, 100, -15],
    # rotation=[0, 0, 0],
    # scale=10.0,            # Adjust as needed for your mesh
    # total_mass=1.0,
    # volume=1.0,
    # color=[1,1,1,1],
    # isStatic=True
    # )
    # add_rigid_object_from_stl(
    # rootNode,  # or rootNode, or wherever you want it
    # name="RigidSphere",
    # stl_path="sphere.stl",
    # translation=[15, 100, -15],
    # rotation=[0, 0, 0],
    # scale=10.0,            # Adjust as needed for your mesh
    # total_mass=1.0,
    # volume=1.0,
    # color=[1,1,1,1],
    # isStatic=True
    # )
    

    return rootNode


