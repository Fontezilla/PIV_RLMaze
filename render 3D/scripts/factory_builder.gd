extends Node3D

const MAP_PATH := "res://data/map_factory.json"
const SUBNODE_MAP_PATH := "res://data/map_subnodes.json"
const SCALE_FACTOR := 0.01
const EDGE_Y := 0.02
const NODE_RADIUS := 0.15

var min_x := INF
var max_x := -INF
var min_z := INF
var max_z := -INF

# node_id -> tipo ("junction", "entry", "exit", "processA_entry", ...) — usado
# pelo episode_replay.gd para saber onde o robot pode rodar de verdade (só em
# junções, ver rules.is_reverse_move no lado Python) vs onde só pode seguir
# em frente ou recuar (entries/exits/estações de processo/corredores).
var node_types: Dictionary = {}


func _ready():
	randomize()

	var data: Dictionary = load_map(MAP_PATH)

	if data.is_empty():
		push_error("Não foi possível carregar o mapa.")
		return

	compute_bounds(data)

	build_floor(data)
	build_nodes(data)
	build_edges(data)
	build_robot_garage()
	build_subnode_markers()

	setup_camera()


# Marcadores invisíveis (sem malha, sem material) para os nós de corredor do
# grafo físico usado pelo lado Python — não têm significado visual nem de
# decisão, só existem para o script de replay conseguir resolver o nome de
# QUALQUER nó (incluindo os de passagem) para uma posição 3D, e assim animar
# o robot a seguir o caminho físico real em vez de saltar direto entre os
# nós "grandes" já desenhados por build_nodes().
func build_subnode_markers():
	var data: Dictionary = load_map(SUBNODE_MAP_PATH)
	if data.is_empty():
		push_error("Não foi possível carregar os sub-nós (" + SUBNODE_MAP_PATH + ").")
		return

	for id in data["nodes"]:
		var node_data: Dictionary = data["nodes"][id]
		var pos := map_to_world(node_data["coords"])

		var marker := Node3D.new()
		marker.name = str(id)
		marker.position = pos
		marker.position.y = EDGE_Y

		add_child(marker)


func build_robot_garage() -> void:
	var n_node := get_node_or_null("N") as Node3D
	if n_node == null:
		push_error("Nao foi possivel criar a garagem dos robos junto ao ponto N.")
		return

	var garage := Node3D.new()
	garage.name = "RobotGarage"
	garage.position = n_node.position + Vector3(1.32, 0.0, 0.0)
	add_child(garage)

	var floor_mat := create_colored_material(Color(0.18, 0.18, 0.18), 0.85)
	var wall_mat := create_colored_material(Color(0.28, 0.28, 0.26), 0.9)
	var roof_mat := create_colored_material(Color(0.12, 0.12, 0.12), 0.9)
	var stripe_mat := create_colored_material(Color(1.0, 0.72, 0.02), 0.5)

	add_box_child(garage, "GarageFloor", Vector3(2.0, 0.1, 1.05), Vector3(0, EDGE_Y + 0.05, 0), floor_mat)
	add_box_child(garage, "GarageBackWall", Vector3(0.1, 0.9, 1.05), Vector3(0.95, EDGE_Y + 0.55, 0), wall_mat)
	add_box_child(garage, "GarageLeftWall", Vector3(2.0, 0.9, 0.1), Vector3(0, EDGE_Y + 0.55, -0.475), wall_mat)
	add_box_child(garage, "GarageRightWall", Vector3(2.0, 0.9, 0.1), Vector3(0, EDGE_Y + 0.55, 0.475), wall_mat)
	add_box_child(garage, "GarageRoof", Vector3(2.1, 0.1, 1.15), Vector3(0, EDGE_Y + 1.05, 0), roof_mat)

	add_box_child(garage, "GarageDoorLine", Vector3(0.04, 0.012, 0.85), Vector3(-0.98, EDGE_Y + 0.106, 0), stripe_mat)


func load_map(path: String) -> Dictionary:
	if not FileAccess.file_exists(path):
		push_error("Ficheiro não encontrado")
		return {}

	var file := FileAccess.open(path, FileAccess.READ)
	var text := file.get_as_text()

	var json := JSON.new()
	json.parse(text)

	return json.data


func map_to_world(coords: Array) -> Vector3:
	return Vector3(
		float(coords[0]) * SCALE_FACTOR,
		0,
		-float(coords[1]) * SCALE_FACTOR
	)


func compute_bounds(data: Dictionary):
	for id in data["nodes"]:
		var node_data: Dictionary = data["nodes"][id]
		var coords: Array = node_data["coords"]

		var p := map_to_world(coords)

		min_x = min(min_x, p.x)
		max_x = max(max_x, p.x)
		min_z = min(min_z, p.z)
		max_z = max(max_z, p.z)


func build_floor(data: Dictionary) -> void:
	var width := (max_x - min_x) + 2.0
	var depth := (max_z - min_z) + 2.0
	var height := 0.5

	var box := BoxMesh.new()
	box.size = Vector3(width, height, depth)

	var floor := MeshInstance3D.new()
	floor.mesh = box

	floor.position = Vector3(
		(min_x + max_x) / 2.0,
		-height / 2.0,
		(min_z + max_z) / 2.0
	)

	var mat: StandardMaterial3D = load("res://textures/WoodFloor051_2K-JPG/WoodFloor051_2K-JPG.tres") as StandardMaterial3D
	if mat != null:
		mat = mat.duplicate() as StandardMaterial3D
		mat.uv1_scale = Vector3(12, 12, 12)
	else:
		mat = StandardMaterial3D.new()
		mat.albedo_color = Color(0.45, 0.32, 0.22)
		mat.roughness = 0.8

	floor.material_override = mat
	add_child(floor)

	var light := DirectionalLight3D.new()
	light.light_energy = 3.0
	light.rotation_degrees = Vector3(-60, 30, 0)
	light.shadow_enabled = true
	add_child(light)


func build_nodes(data: Dictionary):
	for id in data["nodes"]:
		var node_data: Dictionary = data["nodes"][id]
		var node_type: String = node_data["type"]
		node_types[id] = node_type

		var pos := map_to_world(node_data["coords"])

		if node_type != "junction":
			var platform_height := EDGE_Y + 0.1

			var marker := MeshInstance3D.new()
			var box := BoxMesh.new()
			if "process" in node_type:
				box.size = Vector3(0.8, platform_height, 1.2)
			else:
				box.size = Vector3(1.2, platform_height, 0.8)

			marker.mesh = box
			marker.position = pos
			marker.position.y = platform_height / 2.0

			if node_type == "entry":
				marker.position.x += 0.25
			elif node_type == "exit":
				marker.position.x -= 0.25
			elif "process" in node_type:
				marker.position.z += 0.25

			var mat := StandardMaterial3D.new()
			mat.albedo_color = Color(0.2, 0.2, 0.2)
			mat.roughness = 1.0

			marker.material_override = mat
			marker.name = str(id)

			add_child(marker)
			add_safety_border(marker, node_type)

			spawn_zone_doors(marker, node_type)
			if node_type == "exit":
				add_exit_storage_area(marker)

		else:
			var marker := MeshInstance3D.new()

			var cylinder := CylinderMesh.new()
			cylinder.top_radius = NODE_RADIUS * 0.6
			cylinder.bottom_radius = NODE_RADIUS * 0.6
			cylinder.height = 0.02

			marker.mesh = cylinder

			var mat := StandardMaterial3D.new()
			mat.albedo_color = Color(0.8, 0.8, 0.8)

			marker.material_override = mat
			marker.position = pos
			marker.position.y = EDGE_Y + 0.01
			marker.name = str(id)

			add_child(marker)


func add_safety_border(parent_node: Node3D, node_type: String):
	var border_mat: StandardMaterial3D = StandardMaterial3D.new()
	border_mat.albedo_color = get_safety_border_color(node_type)
	border_mat.roughness = 0.55

	var y := 0.066

	if node_type == "entry":
		add_border_strip(parent_node, Vector3(1.08, 0.008, 0.045), Vector3(0, y, -0.365), border_mat)
		add_border_strip(parent_node, Vector3(1.08, 0.008, 0.045), Vector3(0, y, 0.365), border_mat)
		add_border_strip(parent_node, Vector3(0.045, 0.008, 0.68), Vector3(-0.565, y, 0), border_mat)
		add_border_strip(parent_node, Vector3(0.045, 0.008, 0.68), Vector3(0.565, y, 0), border_mat)
	elif node_type == "exit":
		add_border_strip(parent_node, Vector3(1.08, 0.008, 0.045), Vector3(0, y, -0.365), border_mat)
		add_border_strip(parent_node, Vector3(1.08, 0.008, 0.045), Vector3(0, y, 0.365), border_mat)
		add_border_strip(parent_node, Vector3(0.045, 0.008, 0.68), Vector3(-0.565, y, 0), border_mat)
		add_border_strip(parent_node, Vector3(0.045, 0.008, 0.68), Vector3(0.565, y, 0), border_mat)
	elif "entry" in node_type:
		add_border_strip(parent_node, Vector3(0.68, 0.008, 0.045), Vector3(0, y, -0.565), border_mat)
		add_border_strip(parent_node, Vector3(0.68, 0.008, 0.045), Vector3(0, y, 0.565), border_mat)
		add_border_strip(parent_node, Vector3(0.045, 0.008, 1.08), Vector3(-0.365, y, 0), border_mat)
		add_border_strip(parent_node, Vector3(0.045, 0.008, 1.08), Vector3(0.365, y, 0), border_mat)
	elif "exit" in node_type:
		add_border_strip(parent_node, Vector3(0.68, 0.008, 0.045), Vector3(0, y, -0.565), border_mat)
		add_border_strip(parent_node, Vector3(0.68, 0.008, 0.045), Vector3(0, y, 0.565), border_mat)
		add_border_strip(parent_node, Vector3(0.045, 0.008, 1.08), Vector3(-0.365, y, 0), border_mat)
		add_border_strip(parent_node, Vector3(0.045, 0.008, 1.08), Vector3(0.365, y, 0), border_mat)
	else:
		add_border_strip(parent_node, Vector3(1.08, 0.008, 0.045), Vector3(0, y, -0.365), border_mat)
		add_border_strip(parent_node, Vector3(1.08, 0.008, 0.045), Vector3(0, y, 0.365), border_mat)
		add_border_strip(parent_node, Vector3(0.045, 0.008, 0.68), Vector3(-0.565, y, 0), border_mat)
		add_border_strip(parent_node, Vector3(0.045, 0.008, 0.68), Vector3(0.565, y, 0), border_mat)


func get_safety_border_color(node_type: String) -> Color:
	if node_type == "entry":
		return Color(0.05, 0.85, 0.18)
	elif node_type == "exit":
		return Color(0.95, 0.05, 0.03)
	elif "processA" in node_type:
		return Color(1.0, 0.45, 0.02)
	elif "processB" in node_type:
		return Color(0.05, 0.35, 1.0)

	return Color(1.0, 0.72, 0.05)


func add_border_strip(parent_node: Node3D, size: Vector3, local_pos: Vector3, material: Material):
	var strip: MeshInstance3D = MeshInstance3D.new()
	var mesh: BoxMesh = BoxMesh.new()
	mesh.size = size
	strip.mesh = mesh
	strip.position = local_pos
	strip.material_override = material
	parent_node.add_child(strip)


func add_exit_storage_area(parent_node: Node3D) -> void:
	var pad_mat := create_colored_material(Color(0.12, 0.12, 0.12), 0.85)
	var rail_mat := create_colored_material(Color(1.0, 0.72, 0.02), 0.55)

	add_box_child(parent_node, "ExitStoragePad", Vector3(0.54, 0.12, 0.76), Vector3(0.92, 0.0, 0), pad_mat)
	add_box_child(parent_node, "ExitStorageFrontRail", Vector3(0.04, 0.04, 0.76), Vector3(0.67, 0.08, 0), rail_mat)
	add_box_child(parent_node, "ExitStorageBackRail", Vector3(0.04, 0.04, 0.76), Vector3(1.17, 0.08, 0), rail_mat)
	add_box_child(parent_node, "ExitStorageLeftRail", Vector3(0.54, 0.04, 0.04), Vector3(0.92, 0.08, -0.36), rail_mat)
	add_box_child(parent_node, "ExitStorageRightRail", Vector3(0.54, 0.04, 0.04), Vector3(0.92, 0.08, 0.36), rail_mat)


func create_colored_material(color: Color, roughness: float) -> StandardMaterial3D:
	var mat: StandardMaterial3D = StandardMaterial3D.new()
	mat.albedo_color = color
	mat.roughness = roughness
	return mat


func add_box_child(parent_node: Node3D, node_name: String, size: Vector3, local_pos: Vector3, material: Material) -> MeshInstance3D:
	var box_node: MeshInstance3D = MeshInstance3D.new()
	var box_mesh: BoxMesh = BoxMesh.new()
	box_mesh.size = size
	box_node.mesh = box_mesh
	box_node.name = node_name
	box_node.position = local_pos
	box_node.material_override = material
	parent_node.add_child(box_node)
	return box_node


func spawn_zone_doors(parent_node: Node3D, node_type: String):
	if node_type == "entry" or node_type == "exit" or "process" in node_type:
		create_zone_hole(parent_node)
		create_garage_door(parent_node)


func create_zone_hole(parent_node: Node3D) -> MeshInstance3D:
	var hole := MeshInstance3D.new()
	hole.name = "ZoneHole"
	var hole_mesh := BoxMesh.new()
	hole_mesh.size = Vector3(0.44, 0.01, 0.44)
	hole.mesh = hole_mesh
	hole.position = Vector3(0, 0.066, 0)
	hole.visible = false

	var hole_mat := StandardMaterial3D.new()
	hole_mat.albedo_color = Color(0.01, 0.01, 0.01)
	hole_mat.roughness = 1.0
	hole.material_override = hole_mat

	parent_node.add_child(hole)
	return hole


func create_garage_door(parent_node: Node3D) -> MeshInstance3D:
	var garage_door: MeshInstance3D = MeshInstance3D.new()
	garage_door.name = "ZoneDoor"
	var door_mesh: BoxMesh = BoxMesh.new()
	door_mesh.size = Vector3(0.46, 0.006, 0.46)
	garage_door.mesh = door_mesh
	garage_door.position = Vector3(0, 0.068, 0)

	var door_mat: StandardMaterial3D = StandardMaterial3D.new()
	door_mat.albedo_color = Color(0.16, 0.16, 0.16)
	door_mat.roughness = 0.9
	garage_door.material_override = door_mat

	parent_node.add_child(garage_door)
	return garage_door


func build_edges(data: Dictionary):
	var width := 0.7
	var height := 0.1
	var nodes = data["nodes"]

	for id in nodes:
		var node_data = nodes[id]
		if node_data["type"] != "junction":
			continue

		var plate := MeshInstance3D.new()
		var plate_mesh := BoxMesh.new()
		plate_mesh.size = Vector3(width, height, width)
		plate.mesh = plate_mesh
		plate.position = map_to_world(node_data["coords"])
		plate.position.y = EDGE_Y + height / 2.0

		var plate_mat := StandardMaterial3D.new()
		plate_mat.albedo_color = Color(0.2, 0.2, 0.2)
		plate.material_override = plate_mat

		add_child(plate)

	for edge in data["edges"]:
		var a := get_node(edge["from"]) as Node3D
		var b := get_node(edge["to"]) as Node3D

		var start := a.position
		var end := b.position

		var dir := (end - start).normalized()
		var from_node = nodes[edge["from"]]
		var to_node = nodes[edge["to"]]
		var start_offset := 0.0
		var end_offset := 0.0

		if from_node["type"] == "junction":
			start_offset = width / 2.0
		elif from_node["type"] == "entry" or from_node["type"] == "exit":
			start_offset = 0.6
		elif "process" in from_node["type"]:
			start_offset = 0.6

		if to_node["type"] == "junction":
			end_offset = width / 2.0
		elif to_node["type"] == "entry" or to_node["type"] == "exit":
			end_offset = 0.6
		elif "process" in to_node["type"]:
			end_offset = 0.6

		var road_start := start + dir * start_offset
		var road_end := end - dir * end_offset
		var length := road_start.distance_to(road_end)

		if length <= 0.0:
			continue

		var road := MeshInstance3D.new()
		var mesh := BoxMesh.new()
		mesh.size = Vector3(length, height, width)
		road.mesh = mesh

		var mid := (road_start + road_end) / 2.0
		mid.y = EDGE_Y + height / 2.0
		road.position = mid

		var right := dir.cross(Vector3.UP).normalized()

		var basis := Basis()
		basis.x = dir
		basis.y = Vector3.UP
		basis.z = right

		road.transform.basis = basis

		var mat := StandardMaterial3D.new()
		mat.albedo_color = Color(0.2, 0.2, 0.2)
		road.material_override = mat

		add_child(road)

		var line := MeshInstance3D.new()
		var line_mesh := PlaneMesh.new()
		line_mesh.size = Vector2(length, 0.02)
		line.mesh = line_mesh

		line.position = road.position
		line.position.y = road.position.y + height / 2.0 + 0.001
		line.transform.basis = road.transform.basis

		var line_mat := StandardMaterial3D.new()
		line_mat.albedo_color = Color(0.9, 0.9, 0.9)
		line.material_override = line_mat

		add_child(line)

		for side in [-1, 1]:
			var edge_line := MeshInstance3D.new()
			var edge_mesh := PlaneMesh.new()
			edge_mesh.size = Vector2(length, 0.015)
			edge_line.mesh = edge_mesh

			edge_line.position = road.position
			edge_line.position.y = road.position.y + height / 2.0 + 0.001

			var offset_vec = road.transform.basis.z * (width / 2.0 - 0.02) * side
			edge_line.position += offset_vec

			edge_line.transform.basis = road.transform.basis

			var edge_mat := StandardMaterial3D.new()
			edge_mat.albedo_color = Color(0.9, 0.9, 0.9)

			edge_line.material_override = edge_mat
			add_child(edge_line)


func setup_camera():
	var cam := get_node("Camera3D") as Camera3D
	if cam == null:
		return

	var center_x = (min_x + max_x) / 2
	var center_z = (min_z + max_z) / 2
	var center := Vector3(center_x, 0, center_z)

	cam.position = Vector3(center_x, 8, center_z + 8)
	cam.look_at(center, Vector3.UP)

	if cam.has_method("set_focus"):
		cam.call("set_focus", center)
