import Sofa
import Sofa.Core
import Sofa.constants.Key as Key


def createScene(rootNode):
    rootNode.addObject("VisualStyle", displayFlags="showForceFields showBehaviorModels")
    rootNode.addObject("RequiredPlugin", pluginName="SoftRobots SofaPython3")
    # rootNode.gravity.value = [-9810, 0, 0]
    rootNode.addObject("AttachBodyButtonSetting", stiffness=10)
    rootNode.addObject("FreeMotionAnimationLoop")
    rootNode.addObject("GenericConstraintSolver", tolerance=1e-12, maxIterations=10000)

    finger = rootNode.addChild("finger")
    finger.addObject(
        "EulerImplicitSolver", name="odesolver", rayleighStiffness=0.1, rayleighMass=0.1
    )
    finger.addObject("SparseLDLSolver", name="directSolver")
    finger.addObject("MeshGmshLoader", name="loader", filename="LSOVAs/lsovabody.msh")

    finger.addObject("MeshTopology", src="@loader", name="container")
    finger.addObject(
        "MechanicalObject",
        name="tetras",
        template="Vec3",
        showObject=True,
        showObjectScale=1,
    )

    finger.addObject(
        "TetrahedronFEMForceField",
        template="Vec3",
        name="FEM",
        method="large",
        poissonRatio=0.3,
        youngModulus=5,
    )

    finger.addObject("UniformMass", totalMass=0.0008)

    finger.addObject(
        "BoxROI", name="boxROI", box=[-20, -20, -10, 20, 20, 5], drawBoxes=True
    )
    finger.addObject(
        "RestShapeSpringsForceField",
        points="@boxROI.indices",
        stiffness=1e12,
        angularStiffness=1e12,
    )

    finger.addObject("LinearSolverConstraintCorrection")

    cavity = finger.addChild("cavity")
    cavity.addObject(
        "MeshSTLLoader", name="cavityLoader", filename="LSOVAs/lsovavoid.stl"
    )
    cavity.addObject("MeshTopology", src="@cavityLoader", name="cavityMesh")
    cavity.addObject("MechanicalObject", name="cavity")

    cavity.addObject(
        "SurfacePressureConstraint",
        name="SurfacePressureConstraint",
        template="Vec3",
        value=0,
        triangles="@cavityMesh.triangles",
        valueType="pressure",
        flipNormal=True,
    )

    cavity.addObject(
        "BarycentricMapping", name="mapping", mapForces=False, mapMasses=False
    )

    rootNode.addObject(FingerController(rootNode))


class FingerController(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        # print("DEBUG: initialized controller")
        Sofa.Core.Controller.__init__(self, args, kwargs)
        self.node = args[0]
        self.fingerNode = self.node.getChild("finger")
        self.pressureConstraint = self.fingerNode.cavity.getObject(
            "SurfacePressureConstraint"
        )

        self.max_p = 1.5
        self.min_p = -10
        self.delta_p = 0.001

    def onKeypressedEvent(self, e):
        pressureValue = self.pressureConstraint.value.value[0]

        print("DEBUG: current pressure: ", pressureValue)
        print("DEBUG: key pressed: ", e["key"])

        if e["key"] == Key.plus:
            pressureValue += self.delta_p
            if pressureValue > self.max_p:
                pressureValue = self.max_p

        if e["key"] == Key.minus:
            pressureValue -= self.delta_p
            if pressureValue < self.min_p:
                pressureValue = self.min_p

        self.pressureConstraint.value = [pressureValue]
