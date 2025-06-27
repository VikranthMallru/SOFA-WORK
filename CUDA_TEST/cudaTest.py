import Sofa

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
    M1.addObject('MeshVTKLoader', name='volume', filename='mesh/raptorTetra_19409.vtu', onlyAttachedPoints=False)

    # Solvers
    M1.addObject('EulerImplicitSolver', rayleighStiffness=0.1, rayleighMass=0.1)
    M1.addObject('CGLinearSolver', iterations=25, tolerance=1e-6, threshold=1e-20)

    # Topology and mechanics
    M1.addObject('MeshTopology', src="@volume")
    M1.addObject('MechanicalObject', template='CudaVec3f')
    M1.addObject('UniformMass', vertexMass=0.01)

    # Regions of Interest (ROIs)
    M1.addObject('BoxROI', name='box0', box=[-2.2, -1, -10, 2.2, 10, 10], drawBoxes=1)
    M1.addObject('BoxROI', name='box1', box=[-2.2, -1, -1, 2.2, 2.5, 1.5], drawBoxes=1)
    M1.addObject('IndexValueMapper', name="ind_box0", indices="@box0.tetrahedronIndices", value=100000)
    M1.addObject('IndexValueMapper', name="ind_box1", inputValues="@ind_box0.outputValues", indices="@box1.tetrahedronIndices", value=1000000)

    # FEM forcefield with spatially-varying Young's modulus
    M1.addObject('TetrahedronFEMForceField', name="FEM", youngModulus="@ind_box1.outputValues", poissonRatio=0.4, listening=True)

    # Fixed constraint
    M1.addObject('BoxROI', name='box3', box=[-2.2, -0.3, -9.2, 2.2, 0.110668, 2.88584], drawBoxes=1, drawSize=2)
    M1.addObject('FixedConstraint', indices="@box3.indices")

    # Constant force field
    M1.addObject('BoxROI', name='boxF', box=[-2.2, -1, 6.88, 2.2, 10, 10], drawBoxes=True)
    M1.addObject('ConstantForceField', indices="@boxF.indices", force=[7.5, -6.63, -15], showArrowSize=0.1)

    # Plane force field
    M1.addObject('PlaneForceField', normal=[0, 1, 0], d=-0.2, stiffness=100, showPlane=1, showPlaneSize=20)

    # (Optional) Visual models, as in your commented XML, can be added as child nodes here

    return rootNode
