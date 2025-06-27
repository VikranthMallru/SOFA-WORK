# -*- coding: utf-8 -*-
import Sofa.Core
# import Sofa.constants.Key as Key
import Sofa.Simulation
from stlib3.physics.deformable import ElasticMaterialObject
from stlib3.physics.constraints import FixedBox
from softrobots.actuators import PullingCable
from stlib3.physics.collision import CollisionMesh
from splib3.loaders import loadPointListFromFile
from stlib3.scene import MainHeader, ContactHeader
from stlib3.physics.rigid import RigidObject

import sys
import os
import csv
from collections import defaultdict
import numpy as np



def make_roi_boxes(coords, epsilons):
    """Create box definitions for each point and epsilon."""
    return [[x-e, y-e, z-e, x+e, y+e, z+e] for (x, y, z), e in zip(coords, epsilons)]

def add_boxrois(parent_node, roi_boxes):
    """Add BoxROI objects for each box definition."""
    roi_nodes = []
    for idx, box in enumerate(roi_boxes):
        roi = parent_node.addChild(f"ROI_{idx+1}")
        roi.addObject('BoxROI',
                      name=f"roi_{idx+1}",
                      template="Vec3d",
                      box=box,
                      drawBoxes=False,
                      doUpdate=True,
                      strict=False)
        roi_nodes.append(roi)
    return roi_nodes

def add_monitors_to_rois(roi_nodes):
    """Add a Monitor to each ROI node."""
    for idx, roi in enumerate(roi_nodes):
        roi.addObject("Monitor",
            name="roiMonitor",
            template="Vec3d",
            listening=True,
            indices=f"@roi_{idx+1}.indices",
            showPositions=False,
            PositionsColor=[1.0, 0.0, 0.0, 1.0],
            showMinThreshold=0.01,
            showTrajectories=True,
            TrajectoriesPrecision=0.1,
            TrajectoriesColor=[1,1,0,1],
            ExportPositions=False
        )

def add_forcefields_to_rois(roi_nodes, forces):
    """Add a ConstantForceField to each ROI node, with a different force."""
    for roi, force in zip(roi_nodes, forces):
        roi.addObject("ConstantForceField",
                      name="roiConstForce",
                      force=force)

def log_roi_csv(csv_file, cables, roi_nodes, soft_body_node, printInTerminal=1):
    """Log cable, force, and ROI average point data to CSV, print all points and averages in terminal if printInTerminal is set."""
    dofs = soft_body_node.getObject('dofs')
    if dofs is None:
        if printInTerminal:
            print("MechanicalObject 'dofs' not found!")
        return
    positions = dofs.position.value
    avg_roi_coords = []
    for idx, roi in enumerate(roi_nodes):
        boxroi = roi.getObject(f"roi_{idx+1}")
        indices = boxroi.indices.value
        roi_coords = [positions[i] for i in indices]
        if printInTerminal:
            print(f"roi_{idx+1} contains {len(roi_coords)} points:")
        for pt_idx, (x, y, z) in enumerate(roi_coords):
            if printInTerminal:
                print(f"  Point {pt_idx+1}: ({x:.3f}, {y:.3f}, {z:.3f})")
        # Print and compute average for this ROI
        if roi_coords:
            avg_x = sum(pt[0] for pt in roi_coords) / len(roi_coords)
            avg_y = sum(pt[1] for pt in roi_coords) / len(roi_coords)
            avg_z = sum(pt[2] for pt in roi_coords) / len(roi_coords)
            if printInTerminal:
                print(f"  Average: ({avg_x:.3f}, {avg_y:.3f}, {avg_z:.3f})")
            avg_roi_coords += [f"{avg_x:.3f}", f"{avg_y:.3f}", f"{avg_z:.3f}"]
        else:
            if printInTerminal:
                print("  Average: (N/A, N/A, N/A)")
            avg_roi_coords += ["", "", ""]  # Handle empty ROI gracefully

    disp_values = [c.CableConstraint.value[0] for c in cables]
    force_values = [c.CableConstraint.force.value for c in cables]
    file_exists = os.path.isfile(csv_file)
    # Write header only if file doesn't exist or is empty
    if (not file_exists or os.stat(csv_file).st_size == 0) and len(avg_roi_coords) > 0:
        header = ["L1", "L2", "L3", "F1", "F2", "F3"]
        n_boxes = len(avg_roi_coords) // 3
        for i in range(1, n_boxes + 1):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [f"{d:.3f}" for d in disp_values] +
            [f"{f:.3f}" for f in force_values] +
            avg_roi_coords
        )

def spine_log_roi_csv(cables, roi_nodes, soft_body_node, roi_box_centers, spine_csv_file, y_tol=1e-5, printInTerminal=1):
    """
    Log cable, force, and 'spine' points (average of all points in ROI boxes with the same y center) to CSV.
    Also print all points in each y-group and the group average if printInTerminal is set.
    """
    dofs = soft_body_node.getObject('dofs')
    if dofs is None:
        if printInTerminal:
            print("MechanicalObject 'dofs' not found!")
        return
    positions = dofs.position.value

    # --- Group ROI boxes by y value of their center coordinate ---
    y_groups = defaultdict(list)  # key: y value, value: list of (roi_idx, roi_node)
    for idx, center in enumerate(roi_box_centers):
        y_val = center[1]
        # Find if this y value matches any existing group (within tolerance)
        found = False
        for y_key in y_groups:
            if abs(y_val - y_key) < y_tol:
                y_groups[y_key].append(idx)
                found = True
                break
        if not found:
            y_groups[y_val].append(idx)

    # --- For each y-group, collect all points from all ROI boxes in that group ---
    group_points = []
    for y_key in sorted(y_groups.keys()):
        indices_in_group = y_groups[y_key]
        points = []
        if printInTerminal:
            print(f"\nY-group (y={y_key:.3f}), ROI boxes: {indices_in_group}")
        for idx in indices_in_group:
            roi = roi_nodes[idx]
            boxroi = roi.getObject(f"roi_{idx+1}")
            indices = boxroi.indices.value
            roi_coords = [positions[i] for i in indices]
            for pt_idx, (x, y, z) in enumerate(roi_coords):
                if printInTerminal:
                    print(f"  ROI {idx+1} Point {pt_idx+1}: ({x:.3f}, {y:.3f}, {z:.3f})")
            points.extend(roi_coords)
        # Print and compute average for this y-group
        if points:
            avg_x = sum(pt[0] for pt in points) / len(points)
            avg_y = sum(pt[1] for pt in points) / len(points)
            avg_z = sum(pt[2] for pt in points) / len(points)
            if printInTerminal:
                print(f"  Group Average: ({avg_x:.3f}, {avg_y:.3f}, {avg_z:.3f})")
            group_points.append((avg_x, avg_y, avg_z))
        else:
            if printInTerminal:
                print("  Group Average: (N/A, N/A, N/A)")
            group_points.append(("", "", ""))

    # --- Prepare CSV row ---
    disp_values = [c.CableConstraint.value[0] for c in cables]
    force_values = [c.CableConstraint.force.value for c in cables]
    row = [f"{d:.3f}" for d in disp_values] + [f"{f:.3f}" for f in force_values]
    for pt in group_points:
        row += [f"{pt[0]:.3f}" if pt[0] != "" else "",
                f"{pt[1]:.3f}" if pt[1] != "" else "",
                f"{pt[2]:.3f}" if pt[2] != "" else ""]

    # --- Write header if needed ---
    file_exists = os.path.isfile(spine_csv_file)
    if (not file_exists or os.stat(spine_csv_file).st_size == 0) and len(group_points) > 0:
        header = ["L1", "L2", "L3", "F1", "F2", "F3"]
        for i in range(1, len(group_points)+1):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        with open(spine_csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
    # --- Append row ---
    with open(spine_csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

def log_theta_optim_summary(theta, rms_diff, cable_indices=None, forces=None, reset_file=False):
    summary_file = "/home/ci/my_files/TDCR/CSV_Plots/tdcr_theta_optim_summary.csv"
    header = ["theta", "RMS_diff"]
    if cable_indices is not None:
        header += ["cable1_idx", "cable2_idx", "cable3_idx"]
    if forces is not None:
        header += ["F1", "F2", "F3"]
    # If reset_file is True, overwrite file and write header
    if reset_file:
        with open(summary_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
    # Always append the row
    # Convert to float before formatting!
    try:
        theta_f = float(theta)
    except Exception:
        theta_f = 0.0
    try:
        rms_diff_f = float(rms_diff)
    except Exception:
        rms_diff_f = 0.0
    row = [f"{theta_f:.1f}", f"{rms_diff_f:.6f}"]
    if cable_indices is not None:
        row += [int(idx) for idx in cable_indices]
    if forces is not None:
        row += [f"{float(f):.3f}" for f in forces]
    with open(summary_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

def reset_theta_optim_summary():
    summary_file = "/home/ci/my_files/TDCR/CSV_Plots/tdcr_theta_optim_summary.csv"
    header = ["theta", "RMS_diff", "cable1_idx", "cable2_idx", "cable3_idx", "F1", "F2", "F3"]
    with open(summary_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

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
    """
    Adds a rigid object from an STL mesh to the parent_node, using SOFA native components.
    If isStatic is True, the object will be fixed (static) in the scene, like the floor.
    """
    show_object_scale = scale
    if inertia_matrix is None:
        inertia_matrix = [0., 0., 0., 0., 0., 0., 0., 0., 0.]

    rigid = parent_node.addChild(name)

    # Only add solvers and mass if not static
    if not isStatic:
        rigid.addObject('EulerImplicitSolver', name='odesolver')
        rigid.addObject('CGLinearSolver', name='Solver', iterations=25, tolerance=1e-05, threshold=1e-05)
        rigid.addObject('MechanicalObject', name="mstate", template="Rigid3",
                        translation2=translation, rotation2=rotation, showObjectScale=show_object_scale)
        rigid.addObject('UniformMass', name="mass", vertexMass=[total_mass, volume, inertia_matrix[:]])
        rigid.addObject('UncoupledConstraintCorrection')
    else:
        # For static, just add the MechanicalObject (no solver, no mass)
        rigid.addObject('MechanicalObject', name="mstate", template="Rigid3",
                        translation2=translation, rotation2=rotation, showObjectScale=show_object_scale)

    # Collision subnode using STL
    collision = rigid.addChild('collision')
    collision.addObject('MeshSTLLoader', name="loader", filename=stl_path, scale=scale)
    collision.addObject('MeshTopology', src="@loader")
    collision.addObject('MechanicalObject')
    # Set moving=False, simulated=False if static (like your floor)
    # if isStatic:
    collision.addObject('TriangleCollisionModel', moving= not isStatic, simulated=not isStatic)
    collision.addObject('LineCollisionModel', moving=not isStatic, simulated=not isStatic)
    collision.addObject('PointCollisionModel', moving=not isStatic, simulated=not isStatic)
    # else:
    #     collision.addObject('TriangleCollisionModel')
    #     collision.addObject('LineCollisionModel')
    #     collision.addObject('PointCollisionModel')
    collision.addObject('RigidMapping')

    # Visualization subnode using STL
    visu = rigid.addChild("VisualModel")
    visu.loader = visu.addObject('MeshSTLLoader', name="loader", filename=stl_path)
    visu.addObject('OglModel', name="model", src="@loader", scale3d=[show_object_scale]*3, color=color, updateNormals=False)
    visu.addObject('RigidMapping')

    return rigid




class TDCRController(Sofa.Core.Controller):
    def __init__(self,
                 cable_nodes,
                 roi_nodes,
                 soft_body_node,
                 csv_file,
                 roi_box_centers,
                 spine_csv_file,
                 root= None,
                 n_triplets=None,
                 resolution_deg=None,
                 enable_theta_optimization_cables=True,
                 initial_theta_deg=0.0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "TDCRController"
        # self.displacement_step = 0.1
        self.displacement_step = 0.5
        self.pull_keys = {"1": 0, "2": 1, "3": 2}
        self.release_keys = {"!": 0, "@": 1, "#": 2}
        self.cables = cable_nodes
        self.roi_nodes = roi_nodes
        self.soft_body_node = soft_body_node
        self.csv_file = csv_file
        self.roi_box_centers = roi_box_centers
        self.spine_csv_file = spine_csv_file
        self.root = root
        self.n_triplets = n_triplets
        self.resolution_deg = resolution_deg
        self.enable_theta_optimization_cables = enable_theta_optimization_cables
        self.initial_theta_deg = initial_theta_deg
        self.listening = True


        # Clean the CSV file at every restart
        os.makedirs(os.path.dirname(self.csv_file), exist_ok=True)
        with open(self.csv_file, "w", newline="") as f:
            pass
        os.makedirs(os.path.dirname(self.spine_csv_file), exist_ok=True)
        with open(self.spine_csv_file, "w", newline="") as f:
            pass

    def onKeypressedEvent(self, event):

        key = event['key']
        print("pressed")
        # Existing cable control logic
        if key in self.pull_keys:
            idx = self.pull_keys[key]
            self._adjust_cable(idx, self.displacement_step)
        elif key in self.release_keys:
            idx = self.release_keys[key]
            self._adjust_cable(idx, -self.displacement_step)
            if key == "!" and self.cables[0].CableConstraint.value[0] <= 0:
                # self._adjust_cable(0, -self.displacement_step)
                self._adjust_cable(1, self.displacement_step)
                self._adjust_cable(2, self.displacement_step)
                
        elif key == "4":
            for idx in range(3):
                self._adjust_cable(idx, self.displacement_step)
        elif key == "$":
            for idx in range(3):
                self._adjust_cable(idx, -self.displacement_step)
        elif key == "0":
            self.optimize_theta(displacement=15.0)


        # Call the new function for ROI extraction and CSV logging
        # if key != "4" and key != "$":
        log_roi_csv(self.csv_file,
                    self.cables,
                    self.roi_nodes,
                    self.soft_body_node,
                    printInTerminal=0
                    )
        spine_log_roi_csv(
            self.cables,
            self.roi_nodes,
            self.soft_body_node,
            self.roi_box_centers,
            self.spine_csv_file,
            printInTerminal=0
        )


        # if not self.enable_theta_optimization_cables:
            
        # Existing status display
        disp_values = [c.CableConstraint.value[0] for c in self.cables]
        print("Cable displacements: [{}]".format(", ".join(f"{d:.2f}" for d in disp_values)))
        force_values = [c.CableConstraint.force.value for c in self.cables]
        print("Applied forces: [{}]".format(", ".join(f"{f:.2f}" for f in force_values)))

    def _adjust_cable(self, idx, delta):
        current_disp = self.cables[idx].CableConstraint.value[0]
        new_disp = max(0, current_disp + delta)
        self.cables[idx].CableConstraint.value = [new_disp]

    def generate_cable_paths(self, theta0):
        center = (17.5, 0, 17.5)
        radius = 10
        y_values = [i * 10 for i in range(12)]
        thetas = [theta0, theta0 + 2*np.pi/3, theta0 + 4*np.pi/3]
        paths = []
        pull_points = []
        for t in thetas:
            x = center[0] + radius * np.cos(t)
            z = center[2] + radius * np.sin(t)
            path = [[x, y, z] for y in y_values]
            paths.append(path)
            pull_points.append([x, -3, z])
        return paths, pull_points

    def optimize_theta(self, displacement=20.0, settle_steps=20, max_disp=20.0):
        if not self.enable_theta_optimization_cables:
            print("Theta optimization is disabled.")
            return
        reset_theta_optim_summary()
        cables_per_triplet = 3
        best_theta = None
        best_rms_diff = float('inf')
        best_indices = None
        best_forces = None

        print("\nStarting cable angle optimization...")

        thetas = list(range(0, 360, self.resolution_deg))
        if thetas[-1] != 360:
            thetas.append(360)
        for i, theta in enumerate(thetas):
            start = i * 3
            triplet_indices = [start + j for j in range(3)]
            if max(triplet_indices) >= len(self.cables):
                print(f"Skipping theta={theta}° due to insufficient cables for triplet_indices {triplet_indices}")
                continue
            # Release all cables
            for c in self.cables:
                c.CableConstraint.value = [0]
            for _ in range(settle_steps):
                Sofa.Simulation.animate(self.root, 0.01)

            # Contract: Pull only this triplet
            disp = min(displacement, max_disp)
            for idx in triplet_indices:
                self.cables[idx].CableConstraint.value = [disp]
            for _ in range(settle_steps):
                Sofa.Simulation.animate(self.root, 0.01)

            force_values = [self.cables[idx].CableConstraint.force.value for idx in triplet_indices]
            mean_force = sum(force_values) / 3
            rms_diff = np.sqrt(sum((f - mean_force) ** 2 for f in force_values) / 3)

            # Log every value
            log_theta_optim_summary(theta, rms_diff, triplet_indices, force_values)
            triplet_thetas = [(theta + k * 120) % 360 for k in range(3)]
            print(f"thetas: {', '.join(f'{t:.1f}°' for t in triplet_thetas)}, Indices: {triplet_indices}, Forces: {force_values}, RMS_diff: {rms_diff:.6f}")

            if rms_diff < best_rms_diff:
                best_rms_diff = rms_diff
                best_theta = theta
                best_indices = triplet_indices.copy()
                best_forces = force_values.copy()

            # Release this triplet
            for idx in triplet_indices:
                self.cables[idx].CableConstraint.value = [0]
            for _ in range(settle_steps):
                Sofa.Simulation.animate(self.root, 0.01)

        print(f"\nBest theta: {best_theta:.1f}°, Min RMS_diff: {best_rms_diff:.6f}")
   
    @staticmethod
    def rotate_cable_points(points, deg, center=(17.5, 0, 17.5)):
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



def TDCR(parentNode, name="TDCR",
         rotation=[0.0, 0.0, 0.0], translation=[0.0, 0.0, 0.0],
         fixingBox=[36, -1, -1, -1, 6, 36], minForce = -sys.float_info.max, maxForce = sys.float_info.max,
         enable_theta_optimization_cables=True,
         initial_theta_deg=0.0,resolution_deg=5):
    #############################################################################################################
    if 360 % resolution_deg != 0:
        raise ValueError(f"resolution_deg={resolution_deg} is not a divisor of 360. Please provide a value that divides 360 exactly (e.g., 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30, 36, 40, 45, 60, 72, 90, 120, 180, 360).")

    # adding soft body
    tdcr = parentNode.addChild(name)
    #############################################################################################################

    soft_body = ElasticMaterialObject(tdcr,
        volumeMeshFileName="tdcr_volume.vtk",
        surfaceMeshFileName="tdcr_surface.stl",
        collisionMesh="tdcr_collision.stl",
        youngModulus=60000,
        poissonRatio=0.25,
        totalMass=0.03,
        surfaceColor=[0.96, 0.87, 0.70, 1.0],
        rotation=rotation,
        translation=translation,
        withConstraint=False
    )
    tdcr.addChild(soft_body)
    #############################################################################################################
    # rigidBody = RigidObject(tdcr,
    #     surfaceMeshFileName="sphere.stl",
    #     translation=[0, 0, 0],
    #     rotation=[0, 0, 0],
    #     uniformScale=1.0,
    #     color=[0.8, 0.2, 0.2],
    #     isAStaticObject=False
    # )
    # tdcr.addChild(rigidBody)

    #############################################################################################################

    soft_body.addObject('LinearSolverConstraintCorrection')
    # Fix the base
    FixedBox(soft_body, atPositions=fixingBox, doVisualization=True)
    #############################################################################################################
    # --- Define your ROI centers, epsilons, and forces here ---
    c1=loadPointListFromFile("cable1.json")
    c2=loadPointListFromFile("cable2.json")
    c3=loadPointListFromFile("cable3.json")
    # all_points = [c1[-1], c2[-1], c3[-1]]
    all_points = c1 + c2 + c3
    # coords = [c1,c2,c3]   #points
    coords = [
        (17.5, 110, 7.5),    # ROI 1 center
        (8.839, 110, 22.5),  # ROI 2 center
        (26.16, 110, 22.5)   # ROI 3 center
    ]  # ROI centers
    e=3
    epsilons = [e] * len(all_points)                            # epsilons
    forces = [[0,0,0], [0,0,0], [0,0,0]]           # forces, one per ROI

    roi_boxes = make_roi_boxes(all_points, epsilons)
    roi_nodes = add_boxrois(soft_body, roi_boxes)
    add_monitors_to_rois(roi_nodes)
    # add_forcefields_to_rois(roi_nodes, forces)


    #############################################################################################################
    # Instantiating the TDCRController


    output_dir = "/home/ci/my_files/TDCR/CSV_Plots"
    csv_file = os.path.join(output_dir, "tdcr_output.csv")

    roi_box_centers = all_points  # or whatever list you used for ROI centers
    spine_csv_file = os.path.join(output_dir, "tdcr_spine.csv")

    cables = []
    if enable_theta_optimization_cables:
        # --- Generate all triplets around the robot using rotation ---
          # or another divisor of 360
        thetas = list(range(0, 360, resolution_deg))
        if thetas[-1] != 360:
            thetas.append(360)
        n_triplets = len(thetas)
        center = (17.5, 0, 17.5)
        cable_points = [c1, c2, c3]
        all_cables = []
        for step, theta0 in enumerate(thetas):
            for i in range(3):
                rotated_cable = TDCRController.rotate_cable_points(cable_points[i], theta0, center=center)
                pull_point = [rotated_cable[0][0], -3, rotated_cable[0][2]]
                cable = PullingCable(
                    soft_body,
                    f"PullingCable_triplet{step}_cable{i+1}",
                    pullPointLocation=pull_point,
                    rotation=rotation,
                    translation=translation,
                    cableGeometry=rotated_cable
                )
                cable.CableConstraint.minForce = minForce
                cable.CableConstraint.maxForce = maxForce
                all_cables.append(cable)
        cables = all_cables

    else:
        # Always use rotate_cable_points, even if initial_theta_deg == 0
        cable_points = [c1, c2, c3]
        cable_names = ["PullingCable_1", "PullingCable_2", "PullingCable_3"]
        cables = []
        for i in range(3):
            rotated_cable = TDCRController.rotate_cable_points(cable_points[i], initial_theta_deg)
            pull_point = [rotated_cable[0][0], -3, rotated_cable[0][2]]
            cable = PullingCable(
                soft_body,
                cable_names[i],
                pullPointLocation=pull_point,
                rotation=rotation,
                translation=translation,
                cableGeometry=rotated_cable
            )
            cable.CableConstraint.minForce = minForce
            cable.CableConstraint.maxForce = maxForce
            cables.append(cable)
        soft_body.addChild(cable)
        n_triplets = 1


    controller = TDCRController(
        cable_nodes=cables,
        roi_nodes=roi_nodes,
        soft_body_node=soft_body,
        csv_file=csv_file,
        roi_box_centers=roi_box_centers,
        spine_csv_file=spine_csv_file,
        root=parentNode,
        n_triplets=n_triplets,
        resolution_deg=resolution_deg,
        enable_theta_optimization_cables=enable_theta_optimization_cables,
        initial_theta_deg=initial_theta_deg
    )


    soft_body.collisionmodel.TriangleCollisionModel.selfCollision = True 
    soft_body.collisionmodel.LineCollisionModel.selfCollision = True
    soft_body.collisionmodel.PointCollisionModel.selfCollision = True

    soft_body.addObject(controller)


    tdcr.addObject('EulerImplicitSolver', rayleighStiffness=0.1, rayleighMass=0.1)
    #############################################################################################################
    # Loading the collision model
    # Collision model

    # CollisionMesh(soft_body,
    #     name="TDCRCollision",
    #     surfaceMeshFileName="tdcr_collision.stl",
    #     rotation=rotation,
    #     translation=translation,
    #     collisionGroup=1
    # )

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
    rootNode.addObject('RequiredPlugin', name='SofaValidation') # Needed to use components [Monitor]
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.LinearSolver.Iterative') # Needed to use components [CGLinearSolver]  
    rootNode.addObject('RequiredPlugin', name='Sofa.Component.Mapping.NonLinear') # Needed to use components [RigidMapping]
    return

def createScene(rootNode):

    loadRequiredPlugins(rootNode)
    MainHeader(rootNode, gravity=[0.0, -981.0, 0.0],
               plugins=["SoftRobots"])
    ContactHeader(rootNode,
                  alarmDistance=1.0,
                  contactDistance=0.1,
                  frictionCoef=0.08)
    rootNode.bbox = "-50 -50 -50 50 50 50"
    rootNode.VisualStyle.displayFlags = "showVisual showInteractionForceFields"
    TDCR(rootNode,
         enable_theta_optimization_cables=False,
         initial_theta_deg=0.0, resolution_deg=3,
         minForce=0.1)  # Set initial_theta_deg to 0.0 for no rotation

    # add_rigid_object_from_stl(
    # rootNode,  # or rootNode, or wherever you want it
    # name="RigidCube",
    # stl_path="cube.stl",
    # translation=[15, 100, -15],
    # rotation=[0, 0, 0],
    # scale=10.0,            # Adjust as needed for your mesh
    # total_mass=1.0,
    # volume=1.0,
    # color=[1,1,1,1],
    # isStatic=True
    # )
    # add_rigid_object_from_stl(
    # rootNode,  # or rootNode, or wherever you want it
    # name="RigidSphere",
    # stl_path="sphere.stl",
    # translation=[15, 100, -15],
    # rotation=[0, 0, 0],
    # scale=10.0,            # Adjust as needed for your mesh
    # total_mass=1.0,
    # volume=1.0,
    # color=[1,1,1,1],
    # isStatic=True
    # )
    

    return rootNode


