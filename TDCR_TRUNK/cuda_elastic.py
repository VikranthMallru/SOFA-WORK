def CudaElasticMaterialObject(
    attachedTo,
    volumeMeshFileName,
    surfaceMeshFileName=None,
    collisionMesh=None,
    withConstraint=False,
    youngModulus=18000,
    poissonRatio=0.3,
    totalMass=1.0,
    surfaceColor=[0.96, 0.87, 0.70, 1.0],
    rotation=[0.0, 0.0, 0.0],
    translation=[0.0, 0.0, 0.0],
    scale=[1.0, 1.0, 1.0],
    name="CudaElasticMaterialObject"
):
    # Create the main node
    node = attachedTo.addChild(name)

    # Mesh loader
    if volumeMeshFileName.endswith('.msh'):
        loader = node.addObject('MeshGmshLoader', name='loader',
                               filename=volumeMeshFileName,
                               rotation=rotation,
                               translation=translation,
                               scale3d=scale)
    elif volumeMeshFileName.endswith('.gidmsh'):
        loader = node.addObject('GIDMeshLoader', name='loader',
                               filename=volumeMeshFileName,
                               rotation=rotation,
                               translation=translation,
                               scale3d=scale)
    else:
        loader = node.addObject('MeshVTKLoader', name='loader',
                               filename=volumeMeshFileName,
                               rotation=rotation,
                               translation=translation,
                               scale3d=scale)

    # CUDA-based mechanics
    node.addObject('RequiredPlugin', pluginName='SofaCUDA')
    node.addObject('EulerImplicitSolver', name='odesolver')
    node.addObject('CudaCGLinearSolver', name='linearsolver')
    node.addObject('TetrahedronSetTopologyContainer', src='@loader', name='container')
    node.addObject('CudaMechanicalObject', template='CudaVec3f', name='dofs')
    node.addObject('UniformMass', totalMass=totalMass, name='mass')
    node.addObject('CudaTetrahedronFEMForceField', template='CudaVec3f',
                   name='forcefield',
                   poissonRatio=poissonRatio,
                   youngModulus=youngModulus)

    # Optional constraint correction
    if withConstraint:
        node.addObject('LinearSolverConstraintCorrection')

    # Visual model
    if surfaceMeshFileName:
        visNode = node.addChild('VisualModel')
        visNode.addObject('MeshSTLLoader', name='loader', filename=surfaceMeshFileName)
        visNode.addObject('OglModel', src='@loader', color=surfaceColor)
        visNode.addObject('BarycentricMapping', input='@../dofs', output='@dofs')

    # Collision model
    if collisionMesh:
        collisionNode = node.addChild('collisionmodel')
        collisionNode.addObject('MeshSTLLoader', name='loader', filename=collisionMesh)
        collisionNode.addObject('MechanicalObject', src='@loader')
        collisionNode.addObject('TriangleCollisionModel', selfCollision=True)
        collisionNode.addObject('LineCollisionModel', selfCollision=True)
        collisionNode.addObject('PointCollisionModel', selfCollision=True)
        collisionNode.addObject('BarycentricMapping', input='@../dofs', output='@dofs')

    return node
