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
import numpy as np
import os
import csv
# from matplotlib import pyplot as plt

# --- Reset CSV file at script start ---
CSV_LOG_PATH = "/home/ci/my_files/TDCR_6/CSV_Plots/tdcr_log.csv"
with open(CSV_LOG_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["displacement_cable1", "avg_force"])

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
    def __init__(self, cable_nodes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cables = cable_nodes  # List of cable nodes
        self.name = "TDCRController"
        self.displacement_step = 1

        # Key mappings for pull and release
        self.pull_keys = {"1": 0, "2": 1, "3": 2}
        self.release_keys = {"!": 0, "@": 1, "#": 2}

        # Use the global CSV path
        self.log_file = CSV_LOG_PATH

        # --- Reset CSV file when controller is initialized ---
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["displacement_cable1", "avg_force"])

    def onKeypressedEvent(self, event):
        key = event["key"]

        # Pull single cable
        if key in self.pull_keys:
            idx = self.pull_keys[key]
            self._adjust_cable(idx, self.displacement_step)

        # Release single cable
        elif key in self.release_keys:
            idx = self.release_keys[key]
            self._adjust_cable(idx, -self.displacement_step)

        # Contract all cables
        elif key == "4":
            for idx in range(len(self.cables)):
                self._adjust_cable(idx, self.displacement_step)

        # Expand all cables
        elif key == "$":
            for idx in range(len(self.cables)):
                self._adjust_cable(idx, -self.displacement_step)

        disp_values = [c.CableConstraint.value[0] for c in self.cables]
        print("Cable displacements: [{}]".format(", ".join(f"{d:.2f}" for d in disp_values)))
        force_values = [c.CableConstraint.force.value for c in self.cables]
        print("Applied forces: [{}]".format(", ".join(f"{f:.2f}" for f in force_values)))

        # --- CSV Logging ---
        if self.cables:
            disp1 = self.cables[0].CableConstraint.value[0]
            avg_force = sum(force_values) / len(force_values) if force_values else 0.0
            with open(self.log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([disp1, avg_force])

    def _adjust_cable(self, idx, delta):
        current_disp = self.cables[idx].CableConstraint.value[0]
        new_disp = current_disp + delta
        self.cables[idx].CableConstraint.value = [new_disp]

# fixingBox=[36, -1, -1, -1, 6, 36]
def TDCR(parentNode, name="TDCR",
         rotation=[0.0, 0.0, 0.0], translation=[0.0, 0.0, 0.0],
         fixingBox=[-1, -1, -1, 61.52, 11, 61.52], minForce = -sys.float_info.max, maxForce = sys.float_info.max):

    tdcr = parentNode.addChild(name)

    # Deformable object (visual + FEM)
    soft_body = ElasticMaterialObject(tdcr,
        volumeMeshFileName="tdcr_volume.vtk",
        surfaceMeshFileName="tdcr_surface.stl",
        youngModulus=70_000,
        poissonRatio=0.25,
        totalMass=0.03,
        surfaceColor=[0.96, 0.87, 0.70, 1.0],
        rotation=rotation,
        translation=translation
    )
    tdcr.addChild(soft_body)
    # Fix the base
    FixedBox(soft_body, atPositions=fixingBox, doVisualization=True)
    
    roi = soft_body.addChild("ROI")#add roi as child to soft_body(duh)
    # Region of Interest (ROI)
    coord = [ (60.52)/2, 290 , ((60.52-30)/2)+3]
    delta = 1
    roi.addObject("BoxROI",
              name="roi",
              box=[coord[0]-delta, coord[1]-delta, coord[2]-delta, coord[0]+delta, coord[1]+delta, coord[2]+delta],  # (xMin, yMin, zMin, xMax, yMax, zMax)
              position="@../dofs.rest_position",  # MUST point to dofs
              drawBoxes=False,
              doUpdate=True)
    roi.addObject("ConstantForceField",
              name="roiConstForce",
              indices="@roi.indices",
              force=[0, 0, 0])
    
    roi.addObject("Monitor",
    name="roiMonitor",
    template="Vec3d",
    listening=True,
    indices="@roi.indices",
    showPositions=True,
    showVelocities=False,
    showTrajectories=True,
    PositionsColor=[1.0, 0.0, 0.0, 1.0],
    VelocitiesColor=[0.0, 1.0, 0.0, 1.0],
    ForcesColor=[0.0, 0.0, 1.0, 1.0],
    sizeFactor=0.8,
    showMinThreshold=0.001,
    TrajectoriesPrecision=0.1,
    TrajectoriesColor= [1,1,0,1]
    )
    tdcr.addObject('EulerImplicitSolver', rayleighStiffness=0.1, rayleighMass=0.1)

    c1 = rotate_cable_points(loadPointListFromFile("cable1.json"), 90, center=(60.52/2, 0, 60.52/2))
    c2 = rotate_cable_points(loadPointListFromFile("cable1.json"), 270, center=(60.52/2, 0, 60.52/2))
    c3 = rotate_cable_points(loadPointListFromFile("cable1.json"), 240, center=(60.52/2, 0, 60.52/2))

    cable1 = PullingCable(soft_body,
                          "PullingCable_1",
                          pullPointLocation=c1[0],
                          rotation=rotation,
                          translation=translation,
                          cableGeometry=c1)
    cable1.CableConstraint.minForce = minForce
    cable1.CableConstraint.maxForce = maxForce

    cable2 = PullingCable(soft_body,
                          "PullingCable_2",
                          pullPointLocation=c2[0],
                          rotation=rotation,
                          translation=translation,
                          cableGeometry=c2)
    cable2.CableConstraint.minForce = minForce
    cable2.CableConstraint.maxForce = maxForce
    
    # cable3 = PullingCable(soft_body,
    #                       "PullingCable_3",
    #                       pullPointLocation=c3[0],
    #                       rotation=rotation,
    #                       translation=translation,
    #                       cableGeometry=c3)
    # cable3.CableConstraint.minForce = minForce
    # cable3.CableConstraint.maxForce = maxForce

    


    # controller = TDCRController([cable1,cable2,cable3])
    controller = TDCRController([cable1,cable2])
    soft_body.addObject(controller)

    # Collision model
    CollisionMesh(soft_body,
        name="TDCRCollision",
        surfaceMeshFileName="tdcr_collision.stl",
        rotation=rotation,
        translation=translation,
        collisionGroup=1
    )

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
    rootNode.addObject('RequiredPlugin', name='SofaValidation') # Needed to use components [Monitor]
    return

def createScene(rootNode):
    
    loadRequiredPlugins(rootNode)
    MainHeader(rootNode, gravity=[0.0, -981.0, 0.0], 
               plugins=["SoftRobots"])
    ContactHeader(rootNode, 
                  alarmDistance=2.0, 
                  contactDistance=0.5, 
                  frictionCoef=0.08)
    rootNode.bbox = "-50 -50 -50 50 50 50"
    rootNode.VisualStyle.displayFlags = "showVisual showInteractionForceFields"
    TDCR(rootNode,minForce=0.1)
    # TDCR(rootNode)
    return rootNode
