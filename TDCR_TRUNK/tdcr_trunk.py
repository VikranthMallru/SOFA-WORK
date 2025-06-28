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
# from matplotlib import pyplot as plt
# from cuda_elastic import CudaElasticMaterialObject
def rotate_cable_points(points, deg, center=(24.76,0.0,24.76)):
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

class TDCR_trunk_Controller(Sofa.Core.Controller):
    def __init__(self, cable_nodes, roi_node, soft_body_node,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cables = cable_nodes  # List of 2 cable nodes
        self.name = "TDCR_trunk_Controller"
        self.displacement_step = 1
        self.max_displacement = 10000
        self.min_displacement = -10000
        self.roi_node = roi_node
        self.soft_body_node = soft_body_node
        self.roi_points = None

        # Key mappings for pull and release
        self.pull_keys = {"1": 0, "2": 1}
        self.release_keys = {"!": 0, "@": 1}

        # Initialize output directory and CSV file
        self.output_dir = "/home/ci/my_files/TDCR_TRUNK/CSV_Plots"
        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_file = os.path.join(self.output_dir, "tdcr_trunk_output.csv")
        with open(self.csv_file, "w", newline="") as f:
            pass

    def onKeypressedEvent(self, event):
        
        key = event["key"]

        if key in self.pull_keys:
            idx = self.pull_keys[key]
            self._adjust_cable(idx, self.displacement_step)
            # self._adjust_cable(1, -self.displacement_step)
# 
        # Release single cable
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

        # Expand all cables
        elif key == "$":
            for idx in range(len(self.cables)):
                self._adjust_cable(idx, -self.displacement_step)

        disp_values = [c.CableConstraint.value[0] for c in self.cables]
        print("Cable displacements: [{}]".format(", ".join(f"{d:.2f}" for d in disp_values)))
        force_values = [c.CableConstraint.force.value for c in self.cables]
        print("Applied forces: [{}]".format(", ".join(f"{f:.2f}" for f in force_values)))
        self.ROIextractionAndCSVLogging(disp_values, force_values)

    def _adjust_cable(self, idx, delta):
        current_disp = self.cables[idx].CableConstraint.value[0]
        new_disp = current_disp + delta
        self.cables[idx].CableConstraint.value = [new_disp]

    def ROIextractionAndCSVLogging(self, disp_values, force_values):
        if self.roi_points is None:
            self.roi_points = self.roi_node.getObject('roi')
            # print("roi_points initialized:", self.roi_points)
        if self.roi_points:
            indices = self.roi_points.indices.value
            dofs = self.soft_body_node.getObject('dofs')
            if dofs is not None:
                positions = dofs.position.value
                roi_coords = []
                for i in indices:
                    x, y, z = positions[i]
                    roi_coords.append((x, y, z))
                print("ROI points:")
                for idx, (x, y, z) in enumerate(roi_coords):
                    print(f"Point {idx+1}: ({x:.3f}, {y:.3f}, {z:.3f})")
                flat_coords = []
                for (x, y, z) in roi_coords:
                    flat_coords += [f"{x:.3f}", f"{y:.3f}", f"{z:.3f}"]
                file_exists = os.path.isfile(self.csv_file)
                if (not file_exists or os.stat(self.csv_file).st_size == 0) and len(roi_coords) > 0:
                    header = ["L1", "L2", "F1", "F2"]
                    for i in range(len(roi_coords)):
                        header += [f"x{i+1}", f"y{i+1}", f"z{i+1}"]
                    with open(self.csv_file, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(header)
                with open(self.csv_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [f"{d:.3f}" for d in disp_values] +
                        [f"{f:.3f}" for f in force_values] +
                        flat_coords
                    )
            else:
                print("MechanicalObject 'dofs' not found!")
        else:
            print("ROI points engine not initialized!")
# fixingBox=[-1,0,-1,51.52,7,51.52]




def TDCR_trunk(parentNode, name="TDCR_trunk",
         rotation=[0.0, 0.0, 0.0], translation=[0.0, 0.0, 0.0],
         fixingBox=[-1,-4,-1,50.52,10,50.52] , minForce = -sys.float_info.max, maxForce = sys.float_info.max):
# fixingBox=  
    tdcr = parentNode.addChild(name)
# 
    # Deformable object (visual + FEM)
    soft_body = ElasticMaterialObject(tdcr,
        volumeMeshFileName="tdcr_trunk_volume.vtk",
        surfaceMeshFileName="tdcr_trunk_surface.stl",
        collisionMesh="tdcr_trunk_collision.stl",
        withConstraint=False,
        youngModulus=600_000.0,  # Young's modulus in Pascals
        poissonRatio=0.00,
        totalMass=0.115,
        # materialType="NeoHookean",
        surfaceColor=[0.96, 0.87, 0.70, 1.0],
        rotation=rotation,
        translation=translation
    )
    # soft_body = CudaElasticMaterialObject(tdcr,
    #     volumeMeshFileName="tdcr_trunk_volume.vtk",
    #     surfaceMeshFileName="tdcr_trunk_surface.stl",
    #     collisionMesh="tdcr_trunk_collision.stl",
    #     withConstrain=False,
    #     youngModulus=600_000.0,  # Young's modulus in Pascals
    #     poissonRatio=0.00,
    #     totalMass=0.115,
    #     surfaceColor=[0.96, 0.87, 0.70, 1.0],
    #     rotation=rotation,
    #     translation=translation
    # )

    
    # soft_body.collisionmodel.TriangleCollisionModel.selfCollision = True   
    # soft_body.collisionmodel.LineCollisionModel.selfCollision = True
    # soft_body.collisionmodel.PointCollisionModel.selfCollision = True

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
    
    # FORCE APPLICATION
    roi = soft_body.addChild("ROI")#add roi as child to soft_body(duh)
    # Region of Interest (ROI)
    x=24.76
    y=0.0
    z=2.5
    delta = 0.1
    # roi.addObject("BoxROI",
    #           name="roi",
    #           box=[x-delta,y-delta,z-delta,x+delta,y+delta,z+delta],  # (xMin, yMin, zMin, xMax, yMax, zMax)
    #           drawBoxes=True,
    #           doUpdate=True)
    # roi.addObject("ConstantForceField",
    #           name="roiConstForce",
    #         #   indices="@roi.indices",
    #           force=[0, 0, 0])
    
    # roi.addObject("Monitor",
    # name="roiMonitor",
    # template="CudaVec3f",
    # listening=True,
    # indices="@roi.indices",

    # showPositions=True,
    # showVelocities=False,

    # showTrajectories=True,
    # PositionsColor=[1.0, 0.0, 0.0, 1.0],
    # VelocitiesColor=[0.0, 1.0, 0.0, 1.0],
    # ForcesColor=[0.0, 0.0, 1.0, 1.0],
    # sizeFactor=0.8,
    # showMinThreshold=0.01,
    # TrajectoriesPrecision=0.1,
    # TrajectoriesColor= [1,1,0,1]
    # # ExportPath="/home/ci/my_files/TDCR_TRUNK/Plots/roiMonitor_"
    # )




    
    #Add the Cables
    c1 = loadPointListFromFile("cable1.json")
    c1_r = rotate_cable_points(c1, 95)
    # Lower the pull point by 3 units in y direction
    
    # pull1 = [c1_r[0][0], c1_r[0][1] - 3, c1_r[0][2]]
    # cable1 = PullingCable(soft_body,
    #                       "PullingCable_1",
    #                       pullPointLocation=pull1,
    #                       rotation=rotation,
    #                       translation=translation,
    #                       cableGeometry=c1_r)
    # cable1.CableConstraint.minForce = minForce
    # cable1.CableConstraint.maxForce = maxForce

    c2_r = rotate_cable_points(c1, 270)
    # Lower the pull point by 3 units in y direction
    # pull2 = [c2_r[0][0], c2_r[0][1] - 3, c2_r[0][2]]
    # cable2 = PullingCable(soft_body,
    #                       "PullingCable_2",
    #                       pullPointLocation=pull2,
    #                       rotation=rotation,
    #                       translation=translation,
    #                       cableGeometry=c2_r)
    # cable2.CableConstraint.minForce = minForce
    # cable2.CableConstraint.maxForce = maxForce

    


    # controller = TDCR_trunk_Controller([cable1,cable2],roi_node=roi, soft_body_node=soft_body)
    controller = TDCR_trunk_Controller([cable1],roi_node=roi, soft_body_node=soft_body)
    soft_body.addObject(controller)
    tdcr.addObject('EulerImplicitSolver', rayleighStiffness=0.1, rayleighMass=0.1)


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
