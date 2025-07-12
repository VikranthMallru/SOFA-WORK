# -*- coding: utf-8 -*-
import Sofa.Core
import Sofa.constants.Key as Key
from stlib3.physics.deformable import ElasticMaterialObject
from stlib3.physics.constraints import FixedBox
from softrobots.actuators import PullingCable
from stlib3.physics.collision import CollisionMesh
from splib3.loaders import loadPointListFromFile
from stlib3.scene import MainHeader, ContactHeader
import json
import math

def add_cable_tip_roi(soft_body, cable_json, rotation, translation, epsilon=2, roi_name="ROI_tip"):
    with open(cable_json, 'r') as f:
        points = json.load(f)
    last_point = points[-1]
    roi_center = transform_point(last_point, rotation, translation)
    roi_box = [roi_center[0]-epsilon, roi_center[1]-epsilon, roi_center[2]-epsilon,
               roi_center[0]+epsilon, roi_center[1]+epsilon, roi_center[2]+epsilon]
    roi_node = soft_body.addChild(roi_name)
    roi_node.addObject('BoxROI',
                       name=f"{roi_name}_box",
                       template="Vec3d",
                       box=roi_box,
                       drawBoxes=True,
                       doUpdate=True,
                       strict=False)
    return roi_node

def rotate_point(point, rotation):
    rx, ry, rz = [math.radians(a) for a in rotation]
    x, y, z = point
    y1 = y * math.cos(rx) - z * math.sin(rx)
    z1 = y * math.sin(rx) + z * math.cos(rx)
    x1 = x
    z2 = z1 * math.cos(ry) - x1 * math.sin(ry)
    x2 = z1 * math.sin(ry) + x1 * math.cos(ry)
    y2 = y1
    x3 = x2 * math.cos(rz) - y2 * math.sin(rz)
    y3 = x2 * math.sin(rz) + y2 * math.cos(rz)
    z3 = z2
    return [x3, y3, z3]

def transform_point(point, rotation, translation):
    rotated = rotate_point(point, rotation)
    return [rotated[0] + translation[0], rotated[1] + translation[1], rotated[2] + translation[2]]

def get_cable_points_and_pull_point(filename, rotation, translation):
    with open(filename, 'r') as f:
        cable_points = json.load(f)
    base_pull = [cable_points[0][0], cable_points[0][1] - 3, cable_points[0][2]]
    pull_point = transform_point(base_pull, rotation, translation)
    return cable_points, pull_point

def transform_fixed_box(box, rotation, translation):
    p0 = transform_point([box[0], box[1], box[2]], rotation, translation)
    p1 = transform_point([box[3], box[4], box[5]], rotation, translation)
    return [p0[0], p0[1], p0[2], p1[0], p1[1], p1[2]]

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
    show_object_scale = scale
    if inertia_matrix is None:
        inertia_matrix = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
    rigid = parent_node.addChild(name)
    if not isStatic:
        rigid.addObject('EulerImplicitSolver', name='odesolver')
        rigid.addObject('CGLinearSolver', name='Solver', iterations=25, tolerance=1e-05, threshold=1e-05)
        rigid.addObject('MechanicalObject', name="mstate", template="Rigid3",
                        translation2=translation, rotation2=rotation, showObjectScale=show_object_scale)
        rigid.addObject('UniformMass', name="mass", vertexMass=[total_mass, volume, inertia_matrix[:]])
        rigid.addObject('UncoupledConstraintCorrection')
    else:
        rigid.addObject('MechanicalObject', name="mstate", template="Rigid3",
                        translation2=translation, rotation2=rotation, showObjectScale=show_object_scale)
    collision = rigid.addChild('collision')
    collision.addObject('MeshSTLLoader', name="loader", filename=stl_path, scale=scale)
    collision.addObject('MeshTopology', src="@loader")
    collision.addObject('MechanicalObject')
    collision.addObject('TriangleCollisionModel', moving= not isStatic, simulated=not isStatic)
    collision.addObject('LineCollisionModel', moving=not isStatic, simulated=not isStatic)
    collision.addObject('PointCollisionModel', moving=not isStatic, simulated=not isStatic)
    collision.addObject('RigidMapping')
    visu = rigid.addChild("VisualModel")
    visu.loader = visu.addObject('MeshSTLLoader', name="loader", filename=stl_path)
    visu.addObject('OglModel', name="model", src="@loader", scale3d=[show_object_scale]*3, color=color, updateNormals=False)
    visu.addObject('RigidMapping')
    return rigid

def add_soft_object_from_stl(parent_node,
                             name="SoftObject",
                             volume_mesh="tdcr_volume.vtk",
                             surface_mesh="tdcr_surface.stl",
                             collision_mesh="tdcr_collision.stl",
                             young_modulus=60000,
                             poisson_ratio=0.25,
                             total_mass=0.03,
                             surface_color=[0.96, 0.87, 0.70, 1.0],
                             rotation=[0.0, 0.0, 0.0],
                             translation=[0.0, 0.0, 0.0],
                             with_constraint=False,
                             fixing_box=None,
                             scale=[1, 1, 1]):
    soft_obj = ElasticMaterialObject(
        parent_node,
        volumeMeshFileName=volume_mesh,
        surfaceMeshFileName=surface_mesh,
        collisionMesh=collision_mesh,
        youngModulus=young_modulus,
        poissonRatio=poisson_ratio,
        totalMass=total_mass,
        surfaceColor=surface_color,
        rotation=rotation,
        translation=translation,
        withConstraint=with_constraint,
        scale = scale
    )
    parent_node.addChild(soft_obj)
    if fixing_box is not None:
        FixedBox(soft_obj, atPositions=fixing_box, doVisualization=True)
    return soft_obj

class FingerController(Sofa.Core.Controller):
    def __init__(self, cable, soft_body, roi_nodes):
        super().__init__()
        self.cable = cable
        self.soft_body = soft_body
        self.roi_nodes = roi_nodes
        self.name = "FingerController"

    def onKeypressedEvent(self, e):
        displacement = self.cable.CableConstraint.value[0]
        if e["key"] == Key.plus:
            displacement += .1
        elif e["key"] == Key.minus:
            displacement -= .1
            if displacement < 0:
                displacement = 0
        self.cable.CableConstraint.value = [displacement]
        print(f"Cable displacement: {self.cable.CableConstraint.value[0]:.3f}")
        print(f"Applied force: {self.cable.CableConstraint.force.value:.3f}")
        mech_obj = self.soft_body.getObject('dofs')
        forces = mech_obj.force.value
        positions = mech_obj.position.value
        for roi_node in self.roi_nodes:
            roi_name = roi_node.getName()  # FIX: use getName() instead of .name
            boxroi = roi_node.getObject(f"{roi_name}_box")
            if boxroi is None:
                print(f"[ERROR] BoxROI object '{roi_name}_box' not found in node '{roi_name}'")
                continue
            indices = boxroi.indices.value
            print(f"{roi_name} node forces:")
            for i in indices:
                f = forces[i]
                if sum(abs(x) for x in f) > 1e-3:
                    pos = positions[i]
                    print(f"  Node {i}: Force={f}, Pos={pos}")

def Finger(parentNode=None, name="Finger",
           rotation=[0.0, 0.0, 0.0], translation=[0.0, 0.0, 0.0],
           fixingBox=[-15, 0, -3, 5, 3, 17], cable_json="cable.json"):
    cable_points, pull_point = get_cable_points_and_pull_point(cable_json, rotation, translation)
    transformed_box = transform_fixed_box(fixingBox, rotation, translation)
    finger = parentNode.addChild(name)
    soft_body = add_soft_object_from_stl(
        parent_node=parentNode,
        name="SoftObject",
        volume_mesh="finger.vtk",
        surface_mesh="finger.stl",
        collision_mesh="finger.stl",
        young_modulus=80_000,
        poisson_ratio=0.25,
        total_mass=0.03,
        surface_color=[0.96, 0.87, 0.70, 1.0],
        rotation=rotation,
        translation=translation,
        with_constraint=False
    )
    FixedBox(soft_body, atPositions=transformed_box, doVisualization=True)
    cable = PullingCable(
        soft_body,
        "PullingCable",
        pullPointLocation=pull_point,
        rotation=rotation,
        translation=translation,
        cableGeometry=cable_points
    )
    soft_body.collisionmodel.TriangleCollisionModel.selfCollision = True 
    soft_body.collisionmodel.LineCollisionModel.selfCollision = True
    soft_body.collisionmodel.PointCollisionModel.selfCollision = True
    roi_tip = add_cable_tip_roi(soft_body, cable_json, rotation, translation, epsilon=2, roi_name="ROI_tip")
    roi_nodes = [roi_tip]
    soft_body.addObject(FingerController(cable, soft_body, roi_nodes))
    return finger

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
    rootNode.addObject('CollisionPipeline', name='collisionPipeline')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.MechanicalLoad')
    rootNode.addObject('RequiredPlugin', name='SofaValidation')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.LinearSolver.Iterative')
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Mapping.NonLinear')
    return

def createScene(rootNode):
    loadRequiredPlugins(rootNode)
    MainHeader(rootNode, gravity=[0.0, 0.0, 0.0], plugins=["SoftRobots"])
    ContactHeader(rootNode, alarmDistance=4, contactDistance=0.1, frictionCoef=0.5)
    rootNode.VisualStyle.displayFlags = "showBehavior HideVisual"
    Finger(rootNode, name="Finger1", translation=[0.0, 0.0, 0.0], rotation=[0.0, 0.0, 0.0])
    Finger(rootNode, name="Finger2", translation=[60.0, 0.0, 14.0], rotation=[0.0, 180.0, 0.0])
    position = [30,65,9]
    s = 25
    delta = 1
    x, y, z = position
    fixing_box = [x - delta, y - delta, z - delta, x + delta, y + delta, z + delta]
    add_soft_object_from_stl(
        rootNode,
        name="SoftSphere",
        volume_mesh="sphere_volume.vtk",
        surface_mesh="sphere.stl",
        collision_mesh="sphere.stl",
        translation=position,
        rotation=[0, 0, 0],
        young_modulus=600,
        poisson_ratio=0.25,
        total_mass=0.03,
        surface_color=[1, 1, 1, 1],
        with_constraint=False,
        fixing_box=fixing_box,
        scale=[s, s, s]
    )
    return rootNode
