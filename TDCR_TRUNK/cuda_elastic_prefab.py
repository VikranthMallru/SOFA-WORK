import Sofa.Core

class CudaElasticMaterialObject(Sofa.Core.Prefab):
    prefabParameters = [
        {'name': 'volumeMeshFileName', 'type': 'string', 'help': 'Path to the tetrahedral mesh file'},
        {'name': 'name', 'type': 'string', 'help': 'Name of the object', 'default': 'CudaElasticMaterialObject'},
        {'name': 'rotation', 'type': 'vec3', 'help': 'Rotation', 'default': [0.0, 0.0, 0.0]},
        {'name': 'translation', 'type': 'vec3', 'help': 'Translation', 'default': [0.0, 0.0, 0.0]},
        {'name': 'scale', 'type': 'vec3', 'help': 'Scale', 'default': [1.0, 1.0, 1.0]},
        {'name': 'surfaceMeshFileName', 'type': 'string', 'help': 'Path to surface mesh file', 'default': ''},
        {'name': 'poissonRatio', 'type': 'float', 'help': 'Poisson ratio', 'default': 0.3},
        {'name': 'youngModulus', 'type': 'float', 'help': 'Young modulus', 'default': 18000},
        {'name': 'totalMass', 'type': 'float', 'help': 'Total mass', 'default': 1.0}
    ]

    def __init__(self, **kwargs):
        Sofa.Core.Prefab.__init__(self, **kwargs)

    def init(self):
        node = self.parent.addChild(self.name.value)
        # Mesh loader
        if self.volumeMeshFileName.value.endswith('.msh'):
            loader = node.addObject('MeshGmshLoader', name='loader',
                                   filename=self.volumeMeshFileName.value,
                                   rotation=self.rotation.value,
                                   translation=self.translation.value,
                                   scale3d=self.scale.value)
        elif self.volumeMeshFileName.value.endswith('.gidmsh'):
            loader = node.addObject('GIDMeshLoader', name='loader',
                                   filename=self.volumeMeshFileName.value,
                                   rotation=self.rotation.value,
                                   translation=self.translation.value,
                                   scale3d=self.scale.value)
        else:
            loader = node.addObject('MeshVTKLoader', name='loader',
                                   filename=self.volumeMeshFileName.value,
                                   rotation=self.rotation.value,
                                   translation=self.translation.value,
                                   scale3d=self.scale.value)

        # CUDA-based components
        node.addObject('EulerImplicitSolver', name='integration')
        node.addObject('CudaCGLinearSolver', name='solver')  # Use CUDA solver

        node.addObject('TetrahedronSetTopologyContainer', src='@loader', name='container')
        node.addObject('CudaMechanicalObject', template='CudaVec3f', name='dofs')
        node.addObject('UniformMass', totalMass=self.totalMass.value, name='mass')
        node.addObject('CudaTetrahedronFEMForceField', template='CudaVec3f',
                       name='forcefield',
                       poissonRatio=self.poissonRatio.value,
                       youngModulus=self.youngModulus.value)
        # Visual model (optional)
        if self.surfaceMeshFileName.value:
            visNode = node.addChild('VisualModel')
            visNode.addObject('MeshSTLLoader', name='loader', filename=self.surfaceMeshFileName.value)
            visNode.addObject('OglModel', src='@loader')
            visNode.addObject('BarycentricMapping', input='@../dofs', output='@dofs')

        # Return node for further customization
        return node
