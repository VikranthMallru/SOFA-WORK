# # import Sofa
# # from splib3.objectmodel import SofaPrefab, SofaObject
# # # from stlib3.scene import Node # Removed this problematic import
# # from stlib3.visuals import VisualModel

# # @SofaPrefab
# # class CudaElasticMaterialObject(SofaObject):
# #     """
# #     CUDA-accelerated elastic material object for SOFA.
# #     Refactored for stability and corrected SofaObject usage.
# #     Removed direct import of stlib3.scene.Node.
# #     """
# #     def __init__(self,
# #                  parentNode=None, # Renamed from attachedTo for clarity
# #                  volumeMeshFileName=None,
# #                  name="CudaElasticMaterialObject",
# #                  rotation=[0.0, 0.0, 0.0],
# #                  translation=[0.0, 0.0, 0.0],
# #                  scale=[1.0, 1.0, 1.0],
# #                  surfaceMeshFileName=None,
# #                  collisionMesh=None,
# #                  withConstrain=True,
# #                  surfaceColor=[1.0, 1.0, 1.0],
# #                  poissonRatio=0.3,
# #                  youngModulus=18000,
# #                  totalMass=1.0,
# #                  iterative_solver_iterations=50,
# #                  iterative_solver_precision=1e-6):

# #         # Initialize the SofaObject. This makes 'self' the SOFA node.
# #         # It handles parenting to parentNode if provided.
# #         super().__init__(parentNode, name=name)

# #         # Store parameters that might be needed by methods or for clarity
# #         self.volumeMeshFileName = volumeMeshFileName
# #         self.rotation = list(rotation) # Ensure it's a list
# #         self.translation = list(translation) # Ensure it's a list
# #         self.scale = list(scale) # Ensure it's a list
# #         self.surfaceMeshFileName = surfaceMeshFileName
# #         self.collisionMesh = collisionMesh
# #         self.withConstrain = withConstrain
# #         self.surfaceColor = list(surfaceColor) # Ensure it's a list
# #         self.poissonRatio = poissonRatio
# #         self.youngModulus = youngModulus
# #         self.totalMass = totalMass
# #         self.iterative_solver_iterations = iterative_solver_iterations
# #         self.iterative_solver_precision = iterative_solver_precision

# #         # Create the SOFA components within this SofaObject's node (i.e., 'self')
# #         self._createSofaComponents()

# #     def _createSofaComponents(self):
# #         """
# #         Internal method to create and configure SOFA components.
# #         Uses attributes set in __init__.
# #         'self' is the SOFA node where components will be added.
# #         """
# #         if self.volumeMeshFileName is None:
# #             Sofa.msg_error(self, "No volume mesh provided for CudaElasticMaterialObject.")
# #             return

# #         # Load mesh - components are added to 'self' (the SofaObject's node)
# #         # The loader name is 'loader' by convention, used by src='@loader'
# #         if self.volumeMeshFileName.endswith(".msh"):
# #             self.loader = self.createObject('MeshGmshLoader', name='loader', filename=self.volumeMeshFileName, rotation=self.rotation, translation=self.translation, scale3d=self.scale)
# #         elif self.volumeMeshFileName.endswith(".gidmsh"):
# #             self.loader = self.createObject('GIDMeshLoader', name='loader', filename=self.volumeMeshFileName, rotation=self.rotation, translation=self.translation, scale3d=self.scale)
# #         else: # Assuming .vtk or other compatible format
# #             self.loader = self.createObject('MeshVTKLoader', name='loader', filename=self.volumeMeshFileName, rotation=self.rotation, translation=self.translation, scale3d=self.scale)

# #         # CUDA solvers and integration
# #         # These components are added to 'self'
# #         self.integration = self.createObject('EulerImplicitSolver', name='integration') # No specific template needed for solver itself
# #         self.solver = self.createObject('CGLinearSolver', name="solver", iterations=self.iterative_solver_iterations, tolerance=self.iterative_solver_precision, template="CudaVec3f")

# #         # Topology and mechanical objects (CUDA templates)
# #         self.container = self.createObject('TetrahedronSetTopologyContainer', src='@loader', name='container')
# #         self.dofs = self.createObject('MechanicalObject', template='CudaVec3f', name='dofs', position=self.loader.position.getLinkPath()) # Link position for initialization
# #         self.mass = self.createObject('UniformMass', totalMass=self.totalMass, name='mass')

# #         # FEM forcefield with CUDA template
# #         self.forcefield = self.createObject(
# #             'TetrahedronFEMForceField', template='CudaVec3f',
# #             method='large', name='forcefield',
# #             poissonRatio=self.poissonRatio, youngModulus=self.youngModulus
# #         )

# #         # Constraint correction for CUDA
# #         if self.withConstrain:
# #             self.constraintCorrection = self.createObject('UncoupledConstraintCorrection', name="constraintCorrection") # Assign if needed later

# #         # Collision model
# #         if self.collisionMesh:
# #             self.addCollisionModel(self.collisionMesh, self.rotation, self.translation, self.scale)

# #         # Visual model
# #         if self.surfaceMeshFileName:
# #             self.addVisualModel(self.surfaceMeshFileName, self.surfaceColor, self.rotation, self.translation, self.scale)

# #     def addCollisionModel(self, collisionMeshFile, rotation, translation, scale):
# #         # Create a child node for the collision model for better organization.
# #         # 'self' is the main CudaElasticMaterialObject node.
# #         # self.createChild() is available as self is a SofaObject (which is a Node)
# #         self.collision_model_node = self.createChild('CollisionModelNode') # Store as attribute if needed

# #         self.collision_model_node.createObject('MeshSTLLoader', name='loader', filename=collisionMeshFile, rotation=rotation, translation=translation, scale3d=scale)
# #         self.collision_model_node.createObject('TriangleSetTopologyContainer', src='@loader', name='container')
        
# #         # CRITICAL CHANGE: Use Vec3d for collision MechanicalObject unless CUDA collision is specifically intended and configured.
# #         # This MechanicalObject holds the collision surface's DOFs, typically on CPU.
# #         self.collision_model_node.createObject('MechanicalObject', template='Vec3d', name='dofs')
        
# #         self.collision_model_node.createObject('TriangleCollisionModel', name='triangleModel')
# #         self.collision_model_node.createObject('LineCollisionModel', name='lineModel')
# #         self.collision_model_node.createObject('PointCollisionModel', name='pointModel')
        
# #         # Map from the main object's CudaVec3f DOFs to this collision model's Vec3d DOFs.
# #         # '@../dofs' refers to 'self.dofs' (parent node's 'dofs').
# #         # '@./dofs' refers to 'dofs' within 'self.collision_model_node'.
# #         self.collision_model_node.createObject('BarycentricMapping', name='mapping', input='@../dofs', output='@./dofs')

# #     def addVisualModel(self, surfaceMeshFile, color, rotation, translation, scale):
# #         # stlib3.visuals.VisualModel creates its own child node.
# #         # 'parent=self' makes its node a child of CudaElasticMaterialObject's node.
# #         # Store the VisualModel instance if you need to refer to it (e.g., to change color later)
# #         self.visual_model_component = VisualModel(parent=self, 
# #                                                 surfaceMeshFileName=surfaceMeshFile, 
# #                                                 color=color, 
# #                                                 rotation=rotation, 
# #                                                 translation=translation, 
# #                                                 scale3d=scale, # VisualModel often uses scale3d
# #                                                 nameSuffix="_Visu") # Add a suffix to avoid name clashes if main node is also named 'VisualModel'

# #         # The VisualModel helper from stlib3 creates its own MechanicalObject (usually Vec3f).
# #         # We need to map the main simulation's CudaVec3f DOFs to the VisualModel's DOFs.
# #         # visual_model_component.node is the child node created by VisualModel.
# #         # '@../dofs' refers to 'self.dofs'.
# #         # '@./dofs' refers to the 'dofs' within 'self.visual_model_component.node'.
# #         # Note: VisualModel from stlib3 might already create this mapping if input DOFs are specified.
# #         # However, explicitly creating it ensures the connection.
# #         # Check stlib3.VisualModel's implementation; it might create a 'BarycentricMapping' or 'IdentityMapping'.
# #         if hasattr(self.visual_model_component, 'node') and self.visual_model_component.node is not None:
# #             # Ensure the visual model's MechanicalObject is named 'dofs' or adjust link accordingly
# #             # The path '@../../dofs' assumes the mapping is inside visual_model_component.node,
# #             # and visual_model_component.node is a child of self (CudaElasticMaterialObject).
# #             # So, ../ goes to self, and then ../dofs goes to self.dofs.
# #             # A more direct way could be input=self.dofs.getLinkPath()
# #             # output='@./dofs' refers to the 'dofs' in the same node as the mapping (i.e., visual_model_component.node)
# #             self.visual_model_component.node.createObject('BarycentricMapping', name='mapping',
# #                                                          input='@../../dofs', 
# #                                                          output='@./dofs')

# # __all__ = ["CudaElasticMaterialObject"]
# # -*- coding: utf-8 -*-
# import Sofa
# from splib.objectmodel import SofaPrefab, SofaObject
# from stlib.scene import Node
# from stlib.visuals import VisualModel

# @SofaPrefab
# class CudaElasticMaterialObject(SofaObject):
#     """Creates an elastic material object using GPU acceleration (CUDA)."""

#     def __init__(self,
#                  attachedTo=None,
#                  volumeMeshFileName=None,
#                  name="CudaElasticMaterialObject",
#                  rotation=[0.0, 0.0, 0.0],
#                  translation=[0.0, 0.0, 0.0],
#                  scale=[1.0, 1.0, 1.0],
#                  surfaceMeshFileName=None,
#                  collisionMesh=None,
#                  withConstrain=True,
#                  surfaceColor=[1.0, 1.0, 1.0],
#                  poissonRatio=0.3,
#                  youngModulus=18000,
#                  totalMass=1.0,
#                  solver=None):

#         self.node = Node(attachedTo, name)
#         self.createPrefab(volumeMeshFileName, name, rotation, translation, scale,
#                           surfaceMeshFileName, collisionMesh, withConstrain,
#                           surfaceColor, poissonRatio, youngModulus, totalMass, solver)

#     def createPrefab(self,
#                      volumeMeshFileName=None,
#                      name="CudaElasticMaterialObject",
#                      rotation=[0.0, 0.0, 0.0],
#                      translation=[0.0, 0.0, 0.0],
#                      scale=[1.0, 1.0, 1.0],
#                      surfaceMeshFileName=None,
#                      collisionMesh=None,
#                      withConstrain=True,
#                      surfaceColor=[1.0, 1.0, 1.0],
#                      poissonRatio=0.3,
#                      youngModulus=18000,
#                      totalMass=1.0,
#                      solver=None):

#         # Ensure SofaCUDA plugin is loaded
#         if not self.getRoot().getObject("SofaCUDA", warning=False):
#             self.getRoot().createObject("RequiredPlugin", name="SofaCUDA")

#         if self.node is None:
#             Sofa.msg_error("Unable to create CUDA elastic object because it is not attached to any node.")
#             return None

#         if volumeMeshFileName is None:
#             Sofa.msg_error(self.node, "No volume mesh provided.")
#             return None

#         if volumeMeshFileName.endswith(".msh"):
#             self.loader = self.node.createObject('MeshGmshLoader', name='loader', filename=volumeMeshFileName, rotation=rotation, translation=translation, scale3d=scale)
#         elif volumeMeshFileName.endswith(".gidmsh"):
#             self.loader = self.node.createObject('GIDMeshLoader', name='loader', filename=volumeMeshFileName, rotation=rotation, translation=translation, scale3d=scale)
#         else:
#             self.loader = self.node.createObject('MeshVTKLoader', name='loader', filename=volumeMeshFileName, rotation=rotation, translation=translation, scale3d=scale)

#         if solver is None:
#             self.integration = self.node.createObject('EulerImplicitSolver', name='integration')
#             self.solver = self.node.createObject('SparseLDLSolver', name="solver")

#         self.container = self.node.createObject('TetrahedronSetTopologyContainer', src='@loader', name='container')
#         self.dofs = self.node.createObject('MechanicalObject', template='Vec3f', name='dofs')#CudaVec3f

#         self.mass = self.node.createObject('UniformMass', totalMass=totalMass, name='mass')

#         self.forcefield = self.node.createObject('TetrahedronFEMForceField', template='Vec3f',# CudaVec3f
#                                                  method='large', name='forcefield',
#                                                  poissonRatio=poissonRatio, youngModulus=youngModulus)

#         if withConstrain:
#             self.node.createObject('LinearSolverConstraintCorrection', solverName=self.solver.name)

#         if collisionMesh:
#             self.addCollisionModel(collisionMesh, rotation, translation, scale)

#         if surfaceMeshFileName:
#             self.addVisualModel(surfaceMeshFileName, surfaceColor, rotation, translation, scale)

#     def addCollisionModel(self, collisionMesh, rotation=[0.0, 0.0, 0.0], translation=[0.0, 0.0, 0.0], scale=[1., 1., 1.]):
#         self.collisionmodel = self.node.createChild('CollisionModel')
#         self.collisionmodel.createObject('MeshSTLLoader', name='loader', filename=collisionMesh, rotation=rotation, translation=translation, scale3d=scale)
#         self.collisionmodel.createObject('TriangleSetTopologyContainer', src='@loader', name='container')
#         self.collisionmodel.createObject('MechanicalObject', template='Vec3d', name='dofs')
#         self.collisionmodel.createObject('TriangleCollisionModel')
#         self.collisionmodel.createObject('LineCollisionModel')
#         self.collisionmodel.createObject('PointCollisionModel')
#         self.collisionmodel.createObject('BarycentricMapping')

#     def addVisualModel(self, filename, color, rotation, translation, scale=[1., 1., 1.]):
#         self.visualmodel = VisualModel(parent=self.node, surfaceMeshFileName=filename, color=color, rotation=rotation, translation=translation, scale=scale)
#         self.visualmodel.mapping = self.visualmodel.node.createObject('BarycentricMapping', name='mapping')
