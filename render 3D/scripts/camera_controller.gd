extends Camera3D

const MOVE_SPEED := 6.0
const LOOK_SENSITIVITY := 0.005
const MAX_PITCH := deg_to_rad(89.0)
const ZOOM_SPEED_STEP := 1.0
const MIN_SPEED := 1.0
const MAX_SPEED := 40.0

var _speed := MOVE_SPEED
var _yaw := 0.0
var _pitch := 0.0


func _ready():
	# Forca a posicao inicial correta.
	global_transform.origin = Vector3(0, 5, 10)
	look_at(Vector3.ZERO, Vector3.UP)
	_yaw = rotation.y
	_pitch = rotation.x


func set_focus(new_target: Vector3):
	look_at(new_target, Vector3.UP)
	_yaw = rotation.y
	_pitch = rotation.x


func _process(delta):
	var forward := -global_transform.basis.z
	var right := global_transform.basis.x

	var move := Vector3.ZERO
	if Input.is_key_pressed(KEY_W) or Input.is_key_pressed(KEY_UP):
		move += forward
	if Input.is_key_pressed(KEY_S) or Input.is_key_pressed(KEY_DOWN):
		move -= forward
	if Input.is_key_pressed(KEY_D) or Input.is_key_pressed(KEY_RIGHT):
		move += right
	if Input.is_key_pressed(KEY_A) or Input.is_key_pressed(KEY_LEFT):
		move -= right
	if Input.is_key_pressed(KEY_SPACE):
		move += Vector3.UP
	if Input.is_key_pressed(KEY_CTRL):
		move -= Vector3.UP

	if move != Vector3.ZERO:
		global_transform.origin += move.normalized() * _speed * delta


func _input(event):
	# Rodar livremente para qualquer ponto, mantendo o botão direito do rato.
	if event is InputEventMouseMotion and Input.is_mouse_button_pressed(MOUSE_BUTTON_RIGHT):
		_yaw -= event.relative.x * LOOK_SENSITIVITY
		_pitch -= event.relative.y * LOOK_SENSITIVITY
		_pitch = clamp(_pitch, -MAX_PITCH, MAX_PITCH)
		rotation = Vector3(_pitch, _yaw, 0.0)

	# Roda do rato ajusta a velocidade de movimento (não é zoom neste esquema).
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_WHEEL_UP:
			_speed = clamp(_speed + ZOOM_SPEED_STEP, MIN_SPEED, MAX_SPEED)
		if event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			_speed = clamp(_speed - ZOOM_SPEED_STEP, MIN_SPEED, MAX_SPEED)

	# Reset.
	if event is InputEventKey and event.pressed:
		if event.keycode == KEY_F:
			global_transform.origin = Vector3(0, 5, 10)
			look_at(Vector3.ZERO, Vector3.UP)
			_yaw = rotation.y
			_pitch = rotation.x
