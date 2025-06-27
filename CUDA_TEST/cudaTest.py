import Sofa
from stlib3.physics.constraints import FixedBox
def createScene(rootNode):
    # Set simulation timestep
    # rootNode.dt = 0.04

    # Visual style
    rootNode.addObject('VisualStyle', displayFlags="showBehaviorModels showForceFields")

    # Required plugins
    rootNode.addObject('RequiredPlugin', name="Sofa.Component.Engine.Transform")
    rootNode.addObject('RequiredPlugin', name="Sofa.Component.IO.Mesh")
    rootNode.addObject('RequiredPlugin', name="Sofa.Component.LinearSolver.Iterative")
    rootNode.addObject('RequiredPlugin', name="Sofa.Component.ODESolver.Backward")
    rootNode.addObject('RequiredPlugin', name="Sofa.Component.Topology.Container.Constant")
    rootNode.addObject('RequiredPlugin', name="Sofa.Component.Visual")
    rootNode.addObject('RequiredPlugin', name="SofaCUDA")

    # Main simulation node
    M1 = rootNode.addChild("M1")

# Mesh loading
    M1.addObject('MeshVTKLoader', name='volume', filename='/home/ci/my_files/TDCR_TRUNK/tdcr_trunk_volume.vtk', onlyAttachedPoints=False)


    # Solvers
    M1.addObject('EulerImplicitSolver', rayleighStiffness=0.1, rayleighMass=0.1)
    M1.addObject('CGLinearSolver', iterations=25, tolerance=1e-6, threshold=1e-20)

    # Topology and mechanics
    M1.addObject('MeshTopology', src="@volume")
    M1.addObject('MechanicalObject', template='CudaVec3f')
    M1.addObject('UniformMass', vertexMass=0.115)



    # FEM forcefield with spatially-varying Young's modulus
    M1.addObject('TetrahedronFEMForceField', name="FEM", youngModulus=60_000, poissonRatio=0.0, listening=True)
    fixingBox = [-1,-4,-1,50.52,10,50.52]  # Define the fixed box positions
    FixedBox(M1, atPositions=fixingBox, doVisualization=True)


    return rootNode
