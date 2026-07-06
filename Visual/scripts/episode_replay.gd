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
# Correcção fixa de orientação do modelo (o forward do Forklift.glb não
# coincide com o forward assumido pelo look_at do holder). Positivo = roda
# no sentido anti-horário (visto de cima); negativo = horário.
const ROBOT_YAW_OFFSET_DEG := -180.0
# O marcador do nó fica mais baixo que a superfície da estrada — sem isto o
# robot fica meio enterrado no piso. Ajusta este valor até assentar bem.
const ROBOT_HEIGHT_LIFT := 0.150
# Velocidade de rotação suave (rad/s) — quanto maior, mais rápido vira.
const ROBOT_TURN_SPEED := 4.0
# Produto interno entre a direcção actual e a nova: abaixo disto (mais perto
# de -1, oposto) considera-se marcha-atrás e não roda. -0.7 ≈ 135º.
const REVERSE_DOT_THRESHOLD := -0.7
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

var _robot_nodes: Dictionary = {}      # robot_id -> Node3D
var _robot_schedules: Dictionary = {}  # robot_id -> Array
var _robot_leg: Dictionary = {}        # robot_id -> int (indice da paragem actual)

var _box_nodes: Dictionary = {}        # box_id -> MeshInstance3D
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

	_spawn_robots()
	_spawn_boxes()


func _process(delta: float) -> void:
	if _episode.is_empty():
		return

	_sim_time += delta * PLAYBACK_SPEED / TICK_DURATION

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

	for robot_id in _episode["robots"].keys():
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
		holder.visible = false

		if schedule.size() > 0:
			holder.global_position = _robot_position(schedule[0]["node"])


func _spawn_boxes() -> void:
	var box_log: Array = _episode.get("box_log", [])
	if box_log.is_empty():
		return

	var first_snapshot: Array = box_log[0][1]
	for box in first_snapshot:
		var box_id = box["box_id"]

		var mesh := MeshInstance3D.new()
		var box_mesh := BoxMesh.new()
		box_mesh.size = Vector3(BOX_SIZE, BOX_SIZE, BOX_SIZE)
		mesh.mesh = box_mesh

		var mat := StandardMaterial3D.new()
		mat.albedo_color = PIPELINE_COLORS.get(box["pipeline"], Color(0.8, 0.8, 0.8))
		mesh.material_override = mat

		_factory.add_child(mesh)
		_box_nodes[box_id] = mesh

		if box["current_node"] != null:
			mesh.global_position = _node_position(box["current_node"]) + Vector3(0, BOX_LIFT, 0)
		else:
			mesh.visible = false


func _update_robot(robot_id, delta: float) -> void:
	var schedule: Array = _robot_schedules[robot_id]
	var robot: Node3D = _robot_nodes[robot_id]
	if schedule.is_empty():
		return

	if not robot.visible and _sim_time >= schedule[0]["arrival"]:
		robot.visible = true
	if not robot.visible:
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
		new_pos = _robot_position(cur["node"])
		if idx < schedule.size() - 1:
			has_look_target = true
			look_target = _robot_position(schedule[idx + 1]["node"])
	else:
		var nxt: Dictionary = schedule[idx + 1]
		var frac := 1.0
		if nxt["arrival"] > cur["depart"]:
			frac = clamp((_sim_time - cur["depart"]) / (nxt["arrival"] - cur["depart"]), 0.0, 1.0)

		var from_pos := _robot_position(cur["node"])
		var to_pos := _robot_position(nxt["node"])
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


func _apply_event(event: Dictionary) -> void:
	var box_id = event.get("box_id")
	if box_id == null or not _box_nodes.has(box_id):
		return
	var box: Node3D = _box_nodes[box_id]

	match event["type"]:
		"pickup":
			var robot: Node3D = _robot_nodes.get(event["robot_id"])
			if robot == null:
				return
			_reparent(box, robot)
			box.position = Vector3(0, 0.5, 0)
			box.visible = true

		"dropoff":
			_reparent(box, _factory)
			box.global_position = _node_position(event["node"]) + Vector3(0, BOX_LIFT, 0)
			box.visible = true

		"process_start":
			box.visible = false

		"processed":
			_reparent(box, _factory)
			box.global_position = _node_position(event["to_node"]) + Vector3(0, BOX_LIFT, 0)
			box.visible = true


func _reparent(node: Node, new_parent: Node) -> void:
	if node.get_parent() == new_parent:
		return
	node.get_parent().remove_child(node)
	new_parent.add_child(node)
