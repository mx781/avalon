extends Food

class_name Fig

export var destroy_on_impact_velocity_threshold := 2.0


func _on_body_entered(body: Node):
	if (
		not is_edible()
		and _is_impact_velocity_sufficient(body, destroy_on_impact_velocity_threshold)
	):
		hide()
		return
	._on_body_entered(body)
