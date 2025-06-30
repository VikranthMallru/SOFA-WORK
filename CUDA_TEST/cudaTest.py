import Sofa
import Sofa.Core

def createScene(rootNode):
    rootNode.gravity = [0, 0, 0]
    rootNode.dt = 0.02

    # List of required plugins
    plugins = [
        "Sofa.Component.Collision.Detection.Algorithm",
        "Sofa.Component.Collision.Detection.Intersection",
        "Sofa.Component.Collision.Response.Contact",
        "Sofa.Component.IO.Mesh",
        "Sofa.Component.LinearSolver.Iterative",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.Topology.Container.Dynamic",
        "Sofa.GL.Component.Rendering3D",
        "SofaCUDA"
    ]
    for plugin in plugins:
        rootNode.addObject('RequiredPlugin', name=plugin)

    # Collision pipeline setup
    rootNode.addObject('CollisionPipeline', name="CollisionPipeline", verbose=0)
    rootNode.addObject('BruteForceBroadPhase')
    rootNode.addObject('BVHNarrowPhase')
    rootNode.addObject('DefaultContactManager', name="collision response", response="PenalityContactForceField")
    rootNode.addObject('DiscreteIntersection')

    # Path to new mesh files
    mesh_dir = "/home/ci/my_files/TDCR_TRUNK/"

    # Visual mesh loader (STL)
    rootNode.addObject('MeshSTLLoader', name="TrunkSurface", filename=mesh_dir + "tdcr_trunk_surface.stl")

    # Trunk node (formerly Liver)
    trunk = rootNode.addChild('Trunk')
    trunk.gravity = [0, 0, 0]

    # Solvers
    trunk.addObject('EulerImplicitSolver', name="cg_odesolver", rayleighStiffness=0.1, rayleighMass=0.1)
    trunk.addObject('CGLinearSolver', name="linear solver", iterations=25, tolerance=1e-9, threshold=1e-9)

    # Mesh and topology (VTK)
    trunk.addObject('MeshVTKLoader', name="meshLoader", filename=mesh_dir + "tdcr_trunk_volume.vtk")
    trunk.addObject('MechanicalObject', name="dofs", src="@meshLoader", template="CudaVec3f")
    trunk.addObject('TetrahedronSetTopologyContainer', name="topo", src="@meshLoader")
    trunk.addObject('TetrahedronSetGeometryAlgorithms', name="GeomAlgo", template="CudaVec3f")
    trunk.addObject('DiagonalMass', name="DiagonalMass", template="CudaVec3f,CudaVec3f", massDensity=1)
    trunk.addObject('TetrahedronFEMForceField', name="FEM", template="CudaVec3f", method="large",
                    poissonRatio=0.3, youngModulus=60_000, computeGlobalMatrix=0)
    # trunk.addObject('FixedConstraint', name="FixedConstraint", template="CudaVec3f", topology="@topo", indices=[3, 39, 64])

    # Visualization node
    visu = trunk.addChild('Visu')
    visu.tags = ["Visual"]
    visu.gravity = [0, 0, 0]
    visu.addObject('OglModel', name="VisualModel", src="@../../TrunkSurface")
    visu.addObject('BarycentricMapping', name="visual mapping", input="@../dofs", output="@VisualModel")

    # Collision pipeline at root
    rootNode.addObject('CollisionPipeline', name="CollisionPipeline", verbose=0)
    rootNode.addObject('BruteForceBroadPhase')
    rootNode.addObject('BVHNarrowPhase')
    rootNode.addObject('DefaultContactManager', name="collision response", response="PenalityContactForceField")
    rootNode.addObject('DiscreteIntersection')

    # In your trunk node (after loading the mesh and mechanical object)
    trunk.addObject('TriangleCollisionModel')
    trunk.addObject('LineCollisionModel')
    trunk.addObject('PointCollisionModel')

