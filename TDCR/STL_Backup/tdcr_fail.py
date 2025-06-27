# -*- coding: utf-8 -*-
#####################################################################################################################################################
import Sofa
from stlib3.scene import MainHeader, ContactHeader
from stlib3.physics.constraints import FixedBox
from stlib3.scene import ContactHeader
import Sofa.Core
import Sofa.constants.Key as Key
from softrobots.actuators import PullingCable
from splib3.loaders import loadPointListFromFile

class TDCRController(Sofa.Core.Controller): 
    def __init__(self, cables):
        super().__init__()
        self.name = "TDCRController"
        self.cables = cables  # Dictionary: {"1": cable1, "2": cable2, "3": cable3}
        self.pull_keys = {"1": "1", "2": "2", "3": "3"}
        self.release_keys = {"1": "!", "2": "@", "3": "#"}

    def onKeypressedEvent(self, event):
        key = event["key"]

        for k, pull_key in self.pull_keys.items():
            release_key = self.release_keys[k]
            cable = self.cables[k]
            current_disp = cable.CableConstraint.value[0]

            if key == pull_key:
                current_disp += 1.0  # Pull
                cable.CableConstraint.value = [current_disp]
                print(f"[+] Pulled Cable {k} to {current_disp}")

            elif key == release_key:
                current_disp = current_disp - 1.0#max(0.0, current_disp - 1.0)  # Release
                cable.CableConstraint.value = [current_disp]
                print(f"[-] Released Cable {k} to {current_disp}")





def createTDCR(rootNode):
    tdcrNode = rootNode.addChild("TDCR")

    tdcrNode.addObject("EulerImplicitSolver", rayleighStiffness=0.1, rayleighMass=0.1)
    tdcrNode.addObject("SparseLDLSolver")

    
    tdcrNode.addObject("MeshVTKLoader", name="loader", filename="tdcr_volume.vtk")
    tdcrNode.addObject("TetrahedronSetTopologyContainer", src="@loader")
    tdcrNode.addObject("TetrahedronSetGeometryAlgorithms")
    tdcrNode.addObject("MechanicalObject", name="mechObject", template="Vec3d")

    tdcrNode.addObject("TetrahedronFEMForceField", name="fem", 
                       youngModulus=6e4, 
                       poissonRatio=0.25)


    tdcrNode.addObject("UniformMass", totalMass=0.03)
    # 30gms
    # Add a fixed box constraint at the base of the TDCR
    FixedBox(tdcrNode,
             atPositions=[36, -1, -1, -1, 6, 36],  # Adjust this box to match your base
             doVisualization=True)
    # FixedBox(tdcrNode,
    #          atPositions=[0, 0, 0,  8.839, 110, 22.5],  
    #          doVisualization=True)
    visuNode = tdcrNode.addChild("Visual")
    visuNode.addObject("MeshSTLLoader", name="stlLoader", filename="tdcr_surface.stl")
    visuNode.addObject("OglModel", name="oglModel", src="@stlLoader", color=[0.8, 0.2, 0.2])
    visuNode.addObject("BarycentricMapping", input="@../mechObject", output="@oglModel")
    
        # Collision models for self-collision
    collisionNode = tdcrNode.addChild("CollisionModel")
    collisionNode.addObject("MeshSTLLoader", name="loader", filename="tdcr_collision.stl")
    collisionNode.addObject("MeshTopology", src="@loader")
    collisionNode.addObject("MechanicalObject")
    collisionNode.addObject("TriangleCollisionModel")
    collisionNode.addObject("LineCollisionModel")
    collisionNode.addObject("PointCollisionModel")
    collisionNode.addObject("BarycentricMapping", input="@../mechObject", output="@./")


    cable1 = PullingCable(tdcrNode, name="Cable1",
                          pullPointLocation=[17.5 , -3, 7.5],
                          cableGeometry=loadPointListFromFile("cable1.json"))
    # print(loadPointListFromFile("cable1.json"))

    cable2 = PullingCable(tdcrNode, name="Cable2",
                          pullPointLocation=[8.839 , -3, 22.5],
                          cableGeometry=loadPointListFromFile("cable2.json"))

    cable3 = PullingCable(tdcrNode, name="Cable3",
                          pullPointLocation=[26.16 , -3, 22.5],
                          cableGeometry=loadPointListFromFile("cable3.json"))
    cable1.CableConstraint.value = [0.0]
    cable2.CableConstraint.value = [0.0]
    cable3.CableConstraint.value = [0.0]

    # Attach controller for keyboard interaction
    tdcrNode.addObject(TDCRController({
        "1": cable1,
        "2": cable2,
        "3": cable3
    }))

    return tdcrNode

def loadRequiredPlugin(rootNode):
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Collision.Response.Contact')     # DefaultContactManager
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.IO.Mesh')  # MeshSTLLoader, MeshVTKLoader
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.LinearSolver.Direct')  # SparseLDLSolver
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Mass')  # UniformMass
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.ODESolver.Backward')  # EulerImplicitSolver
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.SolidMechanics.FEM.Elastic')  # FEMForceField
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.StateContainer')  # MechanicalObject
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Topology.Container.Dynamic')  # Topology
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Visual')  # VisualStyle
    rootNode.addObject('RequiredPlugin', name='Sofa.GL.Component.Rendering3D')  # OglModel
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Mapping.Linear') # Needed to use components [BarycentricMapping]
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Engine.Select') # Needed to use components [BoxROI]  
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.SolidMechanics.Spring')
    rootNode.addObject('RequiredPlugin', name='SoftRobots') # Needed to use components [CableConstraint]
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Collision.Geometry')       # Triangle, Line, PointCollisionModel
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Collision.Detection.Algorithm')  # BruteForceDetection
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.AnimationLoop') # Needed to use components [FreeMotionAnimationLoop]  
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Collision.Detection.Intersection') # Needed to use components [LocalMinDistance]  
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Constraint.Lagrangian.Solver') # Needed to use components [GenericConstraintSolver]  
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Topology.Container.Constant') # Needed to use components [MeshTopology]

    return

def createScene(rootNode):
    ContactHeader(rootNode, 
                  alarmDistance=4.0, 
                  contactDistance=2.0, 
                  frictionCoef=0.2)
    
    loadRequiredPlugin(rootNode)
   
    # rootNode.addObject("DefaultAnimationLoop")
    # rootNode.addObject("FreeMotionAnimationLoop")
    # rootNode.addObject("GenericConstraintSolver", tolerance="1e-6", maxIterations="1000")

    # redundant??
    # rootNode.addObject("CollisionPipeline", name="collisionPipeline")
    # rootNode.addObject("BruteForceBroadPhase", name="myBroadPhase")
    # rootNode.addObject("BVHNarrowPhase", name="narrowPhase")
    # rootNode.addObject("DefaultContactManager", name="myContactManager", response="FrictionContactConstraint")
    # rootNode.addObject("LocalMinDistance", name="myMinDistance", alarmDistance=4.0, contactDistance=2.0)
    rootNode.addObject("FreeMotionAnimationLoop")
    rootNode.addObject("GenericConstraintSolver", tolerance="1e-6", maxIterations="1000")

    rootNode.addObject("OglSceneFrame", style="2", alignment = "TopRight")
    rootNode.addObject("VisualStyle", displayFlags="showVisual showBehaviorModels showInteractionForceFields")

    rootNode.gravity = [0, -981, 0]
    # rootNode.gravity = [0, 0, 0]
    rootNode.bbox = "-50 -50 -50 50 50 50"  # Adjust bounding box as needed

    createTDCR(rootNode)    

    return rootNode
######################################################################################################################################################

