extends Node3D

## Reproduz um episódio exportado pelo lado Python (scripts/export_demo.py,
## em data/demo_episode.json) — anima os robots ao longo do horário e
## aplica os eventos (pickup/dropoff/process_start/processed) nos
## instantes certos.
##
## Anexar este script a um nó filho de "Factory" (ex. um Node3D vazio
## chamado "EpisodeReplay"), para que já corra DEPOIS de
## factory_builder.gd ter criado os marcadores dos nós (reais + sub-nós).

const EPISODE_PATH := "res://data/demo_episode.json"
const TICK_DURATION := 0.05   # 1 tick = 0.05s de simulação (ver env/render/renderer.py)
const PLAYBACK_SPEED := 1.0   # 1.0 = tempo real; aumenta para acelerar o replay
const BOX_SIZE := 0.25
const BOX_LIFT := 0.3
const STORAGE_BOX_LIFT := 0.19
const BOX_ON_FORKS_OFFSET := Vector3(0.0, 0.22, -0.42)
# Correcção fixa de orientação do modelo (o forward do Forklift.glb não
# coincide com o forward assumido pelo look_at do holder). Positivo = roda
# no sentido anti-horário (visto de cima); negativo = horário.
const ROBOT_YAW_OFFSET_DEG := -180.0
# O marcador do nó fica mais baixo que a superfície da estrada — sem isto o
# robot fica meio enterrado no piso. Ajusta este valor até assentar bem.
const ROBOT_HEIGHT_LIFT := 0.150
# Velocidade de rotação suave (rad/s) — quanto maior, mais rápido vira.
const ROBOT_TURN_SPEED := 4.0
const REVERSE_DOT_THRESHOLD := -0.7
# Produto interno entre a direcção actual e a nova: abaixo disto (mais perto
# de -1, oposto) considera-se marcha-atrás e não roda. -0.7 ≈ 135º.
const GARAGE_OFFSET := Vector3(1.32, 0.0, 0.0)
const GARAGE_ROBOT_SPACING := 0.28
const EXIT_STORAGE_OFFSET := Vector3(0.92, 0.0, 0.0)
const EXIT_STORAGE_COLUMNS := 1
const EXIT_STORAGE_SPACING := 0.28
const ZONE_DOOR_OPEN_TIME := 0.9
const EXIT_BOX_SINK_TIME := 0.45
const EXIT_BOX_TO_STORAGE_DELAY := 0.25
const PROCESS_BOX_SINK_TIME := 0.45
const BOX_VERTICAL_ANIM_DISTANCE := 0.35
const ROBOT_ZONE_STOP_OFFSET := 0.75
# Ajuste manual extra à posição do modelo dentro do holder, já na orientação
# corrigida do modelo (mesma referência que o "forward" dele): X = esquerda(-)/
# direita(+), Y = baixo(-)/cima(+), Z = à frente(-)/atrás(+). Experimenta valores
# pequenos (ex. 0.1, -0.1) e vê o efeito.
const ROBOT_PIVOT_OFFSET := Vector3(0.0, 0.0, 0.5)

const PIPELINE_COLORS := {
	"BLUE":  Color(0.2, 0.55, 0.9),
	"GREEN": Color(0.079, 0.672, 0.0),
	"RED":   Color(0.9, 0.1, 0.2),
}

var _factory: Node3D

var _episode: Dictionary = {}
var _sim_time := 0.0
var _box_log: Array = []
var _box_snapshot_idx := 0

var _robot_nodes: Dictionary = {}      # robot_id -> Node3D
var _robot_schedules: Dictionary = {}  # robot_id -> Array
var _robot_leg: Dictionary = {}        # robot_id -> int (indice da paragem actual)
var _robot_garage_positions: Dictionary = {}  # robot_id -> Vector3

var _box_nodes: Dictionary = {}        # box_id -> MeshInstance3D
var _robot_carrying: Dictionary = {}   # robot_id -> box_id
var _box_carrier: Dictionary = {}      # box_id -> robot_id
var _exit_storage_slots: Dictionary = {}   # box_id -> int
var _exit_storage_counts: Dictionary = {}  # exit node -> int
var _exit_storage_state: Dictionary = {}  # box_id -> "delivering" | "stored"
var _process_entry_state: Dictionary = {}  # box_id -> "entering" | "inside"
var _box_visible_node: Dictionary = {}  # box_id -> node name
var _zone_door_tweens: Dictionary = {}  # node name -> Tween
var _events: Array = []
var _event_idx := 0


func _ready() -> void:
	_factory = get_parent()
	# `_ready()` dos filhos corre ANTES do `_ready()` do pai — o "Factory"
	# (onde factory_builder.gd cria os marcadores dos nós) ainda não
	# terminou de construir o mapa neste ponto. `call_deferred` adia esta
	# inicialização para depois de todos os `_ready()` da cena terminarem.
	call_deferred("_initialize")


func _initialize() -> void:
	var data := _load_json(EPISODE_PATH)
	if data.is_empty():
		push_error("Não foi possível carregar o episódio (" + EPISODE_PATH + ").")
		return
	_episode = data
	_events = _episode.get("events", [])
	_box_log = _episode.get("box_log", [])

	_spawn_robots()
	_spawn_boxes()


func _process(delta: float) -> void:
	if _episode.is_empty():
		return

	_sim_time += delta * PLAYBACK_SPEED / TICK_DURATION

	_process_box_snapshots()

	for robot_id in _robot_schedules.keys():
		_update_robot(robot_id, delta)

	_process_events()


func _load_json(path: String) -> Dictionary:
	if not FileAccess.file_exists(path):
		return {}
	var file := FileAccess.open(path, FileAccess.READ)
	var json := JSON.new()
	if json.parse(file.get_as_text()) != OK:
		return {}
	return json.data


func _node_position(node_name: String) -> Vector3:
	var n := _factory.get_node_or_null(node_name) as Node3D
	if n == null:
		push_error("Nó desconhecido no mapa: " + node_name)
		return Vector3.ZERO
	return n.global_position


func _robot_position(node_name: String) -> Vector3:
	return _node_position(node_name) + Vector3(0, ROBOT_HEIGHT_LIFT, 0)


func _garage_robot_position(robot_index: int, robot_count: int) -> Vector3:
	var center_offset := (float(robot_count) - 1.0) * 0.5
	var lateral_offset := (float(robot_index) - center_offset) * GARAGE_ROBOT_SPACING
	return _node_position("N") + GARAGE_OFFSET + Vector3(0, ROBOT_HEIGHT_LIFT, lateral_offset)


func _schedule_position(robot_id, schedule: Array, leg_index: int) -> Vector3:
	var node_name: String = str(schedule[leg_index]["node"])
	if leg_index == 0 and node_name == "N" and _robot_garage_positions.has(robot_id):
		return _robot_garage_positions[robot_id] as Vector3
	if _is_zone_node(node_name):
		return _zone_approach_position(schedule, leg_index)
	return _robot_position(node_name)


func _is_zone_node(node_name: String) -> bool:
	return node_name.begins_with("entry") or node_name.begins_with("exit") or node_name.begins_with("process")


func _zone_approach_position(schedule: Array, leg_index: int) -> Vector3:
	var node_name: String = str(schedule[leg_index]["node"])
	var zone_pos: Vector3 = _robot_position(node_name)
	var neighbor_pos: Vector3 = Vector3.ZERO
	var has_neighbor: bool = false

	if leg_index > 0:
		neighbor_pos = _robot_position(str(schedule[leg_index - 1]["node"]))
		has_neighbor = true
	elif leg_index < schedule.size() - 1:
		neighbor_pos = _robot_position(str(schedule[leg_index + 1]["node"]))
		has_neighbor = true

	if not has_neighbor:
		return zone_pos

	var approach_dir: Vector3 = neighbor_pos - zone_pos
	approach_dir.y = 0.0
	if approach_dir.length() <= 0.001:
		return zone_pos

	return zone_pos + approach_dir.normalized() * ROBOT_ZONE_STOP_OFFSET


func _is_exit_node(node_name: String) -> bool:
	return node_name.begins_with("exit")


func _is_entry_node(node_name: String) -> bool:
	return node_name.begins_with("entry")


func _is_process_entry_node(node_name: String) -> bool:
	return node_name.begins_with("process") and node_name.ends_with("_entry")


func _box_node_position(box_id, node_name: String) -> Vector3:
	if _is_exit_node(node_name):
		return _exit_storage_position(box_id, node_name)
	return _node_position(node_name) + Vector3(0, BOX_LIFT, 0)


func _exit_storage_position(box_id, exit_node: String) -> Vector3:
	var slot: int = _exit_storage_slot(box_id, exit_node)
	var column: int = slot % EXIT_STORAGE_COLUMNS
	var layer: int = int(slot / EXIT_STORAGE_COLUMNS)
	var centered_column := float(column) - (float(EXIT_STORAGE_COLUMNS) - 1.0) * 0.5

	return _node_position(exit_node) \
		+ EXIT_STORAGE_OFFSET \
		+ Vector3(
			0.0,
			STORAGE_BOX_LIFT + float(layer) * BOX_SIZE,
			centered_column * EXIT_STORAGE_SPACING
		)


func _exit_storage_slot(box_id, exit_node: String) -> int:
	if _exit_storage_slots.has(box_id):
		return int(_exit_storage_slots[box_id])

	var count: int = int(_exit_storage_counts.get(exit_node, 0))
	_exit_storage_counts[exit_node] = count + 1
	_exit_storage_slots[box_id] = count
	return count


func _animate_box_rise(box: Node3D, target_pos: Vector3, duration: float, door_node: String = "") -> Tween:
	if door_node != "":
		_open_zone_door(door_node)
	box.global_position = target_pos - Vector3(0, BOX_VERTICAL_ANIM_DISTANCE, 0)
	box.visible = true
	var tween: Tween = create_tween()
	tween.tween_property(box, "global_position", target_pos, duration)\
		.set_trans(Tween.TRANS_SINE)\
		.set_ease(Tween.EASE_OUT)
	if door_node != "":
		tween.tween_callback(Callable(self, "_close_zone_door").bind(door_node))
	return tween


func _animate_box_sink(box: Node3D, duration: float, door_node: String = "") -> Tween:
	if door_node != "":
		_open_zone_door(door_node)
	var target_pos: Vector3 = box.global_position - Vector3(0, BOX_VERTICAL_ANIM_DISTANCE, 0)
	var tween: Tween = create_tween()
	tween.tween_property(box, "global_position", target_pos, duration)\
		.set_trans(Tween.TRANS_SINE)\
		.set_ease(Tween.EASE_IN)
	tween.tween_callback(Callable(box, "hide"))
	if door_node != "":
		tween.tween_callback(Callable(self, "_close_zone_door").bind(door_node))
	return tween


func _show_box_at_node(box_id, box: Node3D, node_name: String, animate: bool = false) -> void:
	if _is_exit_node(node_name):
		_show_box_at_exit(box_id, box, node_name, animate)
		return

	_release_box(box_id)
	var target_pos: Vector3 = _box_node_position(box_id, node_name)
	if animate:
		_animate_box_rise(box, target_pos, PROCESS_BOX_SINK_TIME, node_name)
	else:
		box.global_position = target_pos
		box.visible = true

	_box_visible_node[box_id] = node_name


func _show_box_enter_process(box_id, box: Node3D, process_node: String) -> void:
	_release_box(box_id)

	var process_state: String = str(_process_entry_state.get(box_id, ""))
	if process_state == "inside":
		box.visible = false
		_box_visible_node[box_id] = process_node
		return
	if process_state == "entering":
		_box_visible_node[box_id] = process_node
		return

	_process_entry_state[box_id] = "entering"
	_box_visible_node[box_id] = process_node
	var process_pos: Vector3 = _box_node_position(box_id, process_node)
	box.global_position = process_pos
	box.visible = true

	var sink_tween: Tween = create_tween()
	sink_tween.tween_interval(0.15)
	sink_tween.tween_callback(Callable(self, "_sink_box_into_process").bind(box_id))


func _mark_box_inside_process(box_id) -> void:
	_process_entry_state[box_id] = "inside"


func _sink_box_into_process(box_id) -> void:
	if not _box_nodes.has(box_id):
		return
	var box: Node3D = _box_nodes[box_id] as Node3D
	var process_node: String = str(_box_visible_node.get(box_id, ""))
	var tween: Tween = _animate_box_sink(box, PROCESS_BOX_SINK_TIME, process_node)
	tween.tween_callback(Callable(self, "_mark_box_inside_process").bind(box_id))


func _show_box_at_exit(box_id, box: Node3D, exit_node: String, animate_arrival: bool = false) -> void:
	_release_box(box_id)

	var storage_state: String = str(_exit_storage_state.get(box_id, ""))
	if storage_state == "stored":
		box.global_position = _exit_storage_position(box_id, exit_node)
		box.visible = true
		_box_visible_node[box_id] = exit_node
		return
	if storage_state == "delivering":
		_box_visible_node[box_id] = exit_node
		return

	_exit_storage_state[box_id] = "delivering"
	_box_visible_node[box_id] = exit_node
	var exit_pos: Vector3 = _node_position(exit_node) + Vector3(0, BOX_LIFT, 0)
	if animate_arrival:
		_animate_box_rise(box, exit_pos, EXIT_BOX_SINK_TIME, exit_node)
	else:
		box.global_position = exit_pos
		box.visible = true

	var sink_tween: Tween = create_tween()
	if animate_arrival:
		sink_tween.tween_interval(EXIT_BOX_SINK_TIME)
	else:
		sink_tween.tween_interval(0.15)
	sink_tween.tween_callback(Callable(self, "_sink_box_before_storage").bind(box_id, exit_node))


func _sink_box_before_storage(box_id, exit_node: String) -> void:
	if not _box_nodes.has(box_id):
		return
	var box: Node3D = _box_nodes[box_id] as Node3D
	var tween: Tween = _animate_box_sink(box, EXIT_BOX_SINK_TIME, exit_node)
	tween.tween_interval(EXIT_BOX_TO_STORAGE_DELAY)
	tween.tween_callback(Callable(self, "_move_box_to_exit_storage").bind(box_id, exit_node))


func _move_box_to_exit_storage(box_id, exit_node: String) -> void:
	if not _box_nodes.has(box_id):
		return
	var box: Node3D = _box_nodes[box_id] as Node3D
	_release_box(box_id)
	_exit_storage_state[box_id] = "stored"
	_animate_box_rise(box, _exit_storage_position(box_id, exit_node), EXIT_BOX_SINK_TIME)


func _open_zone_door(node_name: String) -> void:
	var zone := _factory.get_node_or_null(node_name) as Node3D
	if zone == null:
		return

	var door := zone.get_node_or_null("ZoneDoor") as Node3D
	var hole := zone.get_node_or_null("ZoneHole") as Node3D
	if door == null or hole == null:
		return

	if _zone_door_tweens.has(node_name):
		var old_tween: Tween = _zone_door_tweens[node_name] as Tween
		if old_tween != null:
			old_tween.kill()

	door.visible = false
	hole.visible = true

	_zone_door_tweens.erase(node_name)


func _pulse_zone_door(node_name: String, open_time: float = ZONE_DOOR_OPEN_TIME) -> void:
	_open_zone_door(node_name)
	var tween: Tween = create_tween()
	_zone_door_tweens[node_name] = tween
	tween.tween_interval(open_time)
	tween.tween_callback(Callable(self, "_close_zone_door").bind(node_name))


func _close_zone_door(node_name: String) -> void:
	var zone := _factory.get_node_or_null(node_name) as Node3D
	if zone == null:
		return

	var door := zone.get_node_or_null("ZoneDoor") as Node3D
	var hole := zone.get_node_or_null("ZoneHole") as Node3D
	if door != null:
		door.visible = true
	if hole != null:
		hole.visible = false
	_zone_door_tweens.erase(node_name)


func _find_first_mesh(node: Node) -> MeshInstance3D:
	if node is MeshInstance3D:
		return node
	for child in node.get_children():
		var found := _find_first_mesh(child)
		if found != null:
			return found
	return null


func _spawn_robots() -> void:
	var forklift_scene: PackedScene = load("res://models/Forklift.glb")
	var robot_ids: Array = (_episode["robots"] as Dictionary).keys()
	var robot_count: int = robot_ids.size()
	var robot_index: int = 0

	for robot_id in robot_ids:
		var schedule: Array = _episode["robots"][robot_id]
		_robot_schedules[robot_id] = schedule
		_robot_leg[robot_id] = 0

		var holder := Node3D.new()
		holder.name = "Robot_" + str(robot_id)
		_factory.add_child(holder)

		var robot := forklift_scene.instantiate()
		holder.add_child(robot)
		robot.scale = Vector3(0.25, 0.25, 0.25)
		robot.rotation.y = deg_to_rad(ROBOT_YAW_OFFSET_DEG)

		# A malha real (2 níveis dentro do modelo, ex. "Cylinder_002") tem um
		# pivô deslocado ~54 unidades embutido no import — cancela-se
		# aplicando o deslocamento inverso, já passado pela mesma
		# rotação/escala do nó raiz, para a malha acabar centrada no holder.
		var mesh := _find_first_mesh(robot)
		if mesh != null:
			robot.position = -(robot.basis * mesh.position) + robot.basis * ROBOT_PIVOT_OFFSET

		_robot_nodes[robot_id] = holder
		holder.visible = true

		if schedule.size() > 0:
			var garage_position: Vector3 = _garage_robot_position(robot_index, robot_count)
			_robot_garage_positions[robot_id] = garage_position
			holder.global_position = garage_position

		robot_index += 1


func _spawn_boxes() -> void:
	if _box_log.is_empty():
		return

	for snapshot in _box_log:
		for box in snapshot[1]:
			_ensure_box_node(box)

	if _box_log.size() > 0:
		_apply_box_snapshot(_box_log[0][1])
		_box_snapshot_idx = 1


func _ensure_box_node(box_state: Dictionary) -> Node3D:
	var box_id = box_state["box_id"]
	if _box_nodes.has(box_id):
		return _box_nodes[box_id] as Node3D

	var mesh := MeshInstance3D.new()
	var box_mesh := BoxMesh.new()
	box_mesh.size = Vector3(BOX_SIZE, BOX_SIZE, BOX_SIZE)
	mesh.mesh = box_mesh

	var mat := StandardMaterial3D.new()
	mat.albedo_color = PIPELINE_COLORS.get(box_state["pipeline"], Color(0.8, 0.8, 0.8))
	mesh.material_override = mat
	mesh.visible = false

	_factory.add_child(mesh)
	_box_nodes[box_id] = mesh
	return mesh


func _update_robot(robot_id, delta: float) -> void:
	var schedule: Array = _robot_schedules[robot_id]
	var robot: Node3D = _robot_nodes[robot_id]
	if schedule.is_empty():
		return

	if _sim_time < schedule[0]["arrival"]:
		return

	var idx: int = _robot_leg[robot_id]
	while idx < schedule.size() - 1 and _sim_time >= schedule[idx + 1]["arrival"]:
		idx += 1
	_robot_leg[robot_id] = idx

	var cur: Dictionary = schedule[idx]
	var new_pos: Vector3
	var has_look_target := false
	var look_target := Vector3.ZERO

	if _sim_time <= cur["depart"] or idx >= schedule.size() - 1:
		# Parado em `cur` — se houver um próximo troço, já começa a virar
		# suavemente para essa direcção antes de arrancar (em vez de saltar
		# a rotação de repente quando o movimento começa).
		new_pos = _schedule_position(robot_id, schedule, idx)
		if idx < schedule.size() - 1:
			has_look_target = true
			look_target = _schedule_position(robot_id, schedule, idx + 1)
	else:
		var nxt: Dictionary = schedule[idx + 1]
		var frac := 1.0
		if nxt["arrival"] > cur["depart"]:
			frac = clamp((_sim_time - cur["depart"]) / (nxt["arrival"] - cur["depart"]), 0.0, 1.0)

		var from_pos := _schedule_position(robot_id, schedule, idx)
		var to_pos := _schedule_position(robot_id, schedule, idx + 1)
		new_pos = from_pos.lerp(to_pos, frac)
		has_look_target = true
		look_target = to_pos

	robot.global_position = new_pos

	if has_look_target and look_target.distance_to(robot.global_position) > 0.001:
		var dir := (look_target - robot.global_position).normalized()
		var current_forward := -robot.global_transform.basis.z.normalized()

		# Se a direcção a seguir for quase oposta à que já está virado, é
		# uma marcha-atrás (ex. sair de um beco sem saída, como uma estação
		# de processo) — mantém a orientação e não roda, só anda para trás.
		if current_forward.dot(dir) > REVERSE_DOT_THRESHOLD:
			var target_quat := Quaternion(Basis.looking_at(dir, Vector3.UP))
			var t: float = clamp(ROBOT_TURN_SPEED * delta, 0.0, 1.0)
			robot.quaternion = robot.quaternion.slerp(target_quat, t)


func _process_events() -> void:
	while _event_idx < _events.size() and _events[_event_idx]["t"] <= _sim_time:
		_apply_event(_events[_event_idx])
		_event_idx += 1


func _process_box_snapshots() -> void:
	while _box_snapshot_idx < _box_log.size() and _box_log[_box_snapshot_idx][0] <= _sim_time:
		_apply_box_snapshot(_box_log[_box_snapshot_idx][1])
		_box_snapshot_idx += 1


func _apply_box_snapshot(snapshot: Array) -> void:
	for box_state in snapshot:
		var box_id = box_state["box_id"]
		var box: Node3D = _ensure_box_node(box_state)
		var carrier = box_state.get("carried_by")
		var current_node = box_state.get("current_node")
		var status: String = str(box_state.get("status", ""))

		if current_node != null:
			var node_name: String = str(current_node)
			if _is_process_entry_node(node_name) and status == "PROCESSING":
				_show_box_enter_process(box_id, box, node_name)
			else:
				var was_at_node := str(_box_visible_node.get(box_id, "")) == node_name
				var animate_entry_spawn := _box_snapshot_idx > 0 and _is_entry_node(node_name) and not was_at_node
				_show_box_at_node(box_id, box, node_name, animate_entry_spawn)
		elif carrier != null and _robot_nodes.has(carrier):
			_release_box(box_id)
			_release_robot_box(carrier)
			_process_entry_state.erase(box_id)
			_box_visible_node.erase(box_id)
			var robot: Node3D = _robot_nodes[carrier] as Node3D
			_reparent_keep_global(box, robot)
			box.position = BOX_ON_FORKS_OFFSET
			box.visible = true
			_robot_carrying[carrier] = box_id
			_box_carrier[box_id] = carrier
		else:
			_release_box(box_id)
			_process_entry_state.erase(box_id)
			_box_visible_node.erase(box_id)
			box.visible = false


func _apply_event(event: Dictionary) -> void:
	var box_id = event.get("box_id")
	if box_id == null or not _box_nodes.has(box_id):
		return
	var box: Node3D = _box_nodes[box_id]

	match event["type"]:
		"pickup":
			if not _box_log.is_empty():
				return
			var robot: Node3D = _robot_nodes.get(event["robot_id"])
			if robot == null:
				return
			_release_robot_box(event["robot_id"])
			_release_box(box_id)
			_reparent_keep_global(box, robot)
			box.position = BOX_ON_FORKS_OFFSET
			box.visible = true
			_robot_carrying[event["robot_id"]] = box_id
			_box_carrier[box_id] = event["robot_id"]

		"dropoff":
			var drop_node: String = str(event["node"])
			if _is_process_entry_node(drop_node) and not bool(event.get("queued", false)):
				_show_box_enter_process(box_id, box, drop_node)
			else:
				_show_box_at_node(box_id, box, drop_node, false)

		"process_start":
			_release_box(box_id)
			if event.has("node"):
				var node_name: String = str(event["node"])
				_show_box_enter_process(box_id, box, node_name)

		"processed":
			_process_entry_state.erase(box_id)
			_show_box_at_node(box_id, box, str(event["to_node"]), true)


func _release_robot_box(robot_id) -> void:
	if not _robot_carrying.has(robot_id):
		return
	_release_box(_robot_carrying[robot_id])


func _release_box(box_id) -> void:
	if not _box_nodes.has(box_id):
		return

	if _box_carrier.has(box_id):
		var robot_id = _box_carrier[box_id]
		_box_carrier.erase(box_id)
		if _robot_carrying.get(robot_id) == box_id:
			_robot_carrying.erase(robot_id)

	var box: Node3D = _box_nodes[box_id]
	_reparent_keep_global(box, _factory)


func _reparent_keep_global(node: Node3D, new_parent: Node) -> void:
	if node.get_parent() == new_parent:
		return
	var old_transform := node.global_transform
	node.get_parent().remove_child(node)
	new_parent.add_child(node)
	node.global_transform = old_transform
