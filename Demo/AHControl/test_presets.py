import argparse
import math
import sys
import time
import tomllib

from rustypot import Scs0009PyController


PRESETS = {
    "zero": {},
    "open-small": {
        "r_finger1": (-1, 1),
        "r_finger2": (-1, 1),
        "r_finger3": (-1, 1),
        "r_finger4": (-1, 1),
    },
    "close-small": {
        "r_finger1": (1, -1),
        "r_finger2": (1, -1),
        "r_finger3": (1, -1),
        "r_finger4": (1, -1),
    },
    "index-close": {"r_finger1": (1, -1)},
    "middle-close": {"r_finger2": (1, -1)},
    "ring-close": {"r_finger3": (1, -1)},
    "thumb-close": {"r_finger4": (1, -1)},
}


SEQUENCE = [
    "zero",
    "index-close",
    "zero",
    "middle-close",
    "zero",
    "ring-close",
    "zero",
    "thumb-close",
    "zero",
    "close-small",
    "zero",
    "open-small",
    "zero",
]


def load_config(path):
    with open(path, "rb") as f:
        config = tomllib.load(f)

    motors = config.get("motors")
    if motors is None:
        motors = config.get("Fingers", {}).get("motors")
    if motors is None:
        raise KeyError("Could not find a 'motors' array in the config file")

    for finger in motors:
        if finger["motor1"]["model"] != "SCS0009" or finger["motor2"]["model"] != "SCS0009":
            raise ValueError("Only SCS0009 motors are supported")
    return motors


def motor_lists(motors):
    ids = []
    offsets = []
    for finger in motors:
        ids.extend([finger["motor1"]["id"], finger["motor2"]["id"]])
        offsets.extend([finger["motor1"]["offset"], finger["motor2"]["offset"]])
    return ids, offsets


def goals_for_preset(motors, preset_name, amplitude_rad):
    preset = PRESETS[preset_name]
    ids = []
    goals = []

    for finger in motors:
        name = finger["finger_name"]
        scale1, scale2 = preset.get(name, (0, 0))

        for motor_key, scale in (("motor1", scale1), ("motor2", scale2)):
            motor = finger[motor_key]
            goal = motor["offset"] + scale * amplitude_rad
            if motor["invert"]:
                goal = -goal
            ids.append(motor["id"])
            goals.append(goal)

    return ids, goals


def move(controller, motors, preset_name, amplitude_rad):
    ids, goals = goals_for_preset(motors, preset_name, amplitude_rad)
    controller.sync_write_goal_position(ids, goals)


def print_positions(controller, ids):
    try:
        positions = controller.sync_read_present_position(ids)
    except Exception as exc:
        print(f"Could not read positions: {exc}", file=sys.stderr)
        return

    formatted = ", ".join(f"{motor_id}:{pos[0]:+.3f}" for motor_id, pos in zip(ids, positions))
    print(f"present positions rad: {formatted}")


def wait_for_user(enabled, message):
    if enabled:
        input(f"{message} Press Enter to continue, Ctrl+C to stop. ")
    else:
        print(message)


def main():
    parser = argparse.ArgumentParser(description="Safely test Amazing Hand preset motor motions.")
    parser.add_argument("--serialport", default="/dev/ttyACM0")
    parser.add_argument("--config", default="AHControl/config/r_hand.toml")
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument("--amplitude-deg", type=float, default=12.0)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--hold", type=float, default=1.0)
    parser.add_argument("--preset", choices=[*PRESETS.keys(), "sequence"], default="sequence")
    parser.add_argument("--no-prompt", action="store_true")
    parser.add_argument("--keep-torque", action="store_true")
    args = parser.parse_args()

    motors = load_config(args.config)
    motor_ids, _ = motor_lists(motors)
    amplitude_rad = math.radians(args.amplitude_deg)
    prompt = not args.no_prompt

    print(f"Opening {args.serialport} at {args.baudrate}")
    print(f"Using config {args.config}")
    print(f"Testing amplitude: {args.amplitude_deg:.1f} deg ({amplitude_rad:.3f} rad)")

    controller = Scs0009PyController(
        serial_port=args.serialport,
        baudrate=args.baudrate,
        timeout=0.5,
    )

    try:
        controller.sync_write_goal_speed(motor_ids, [args.speed] * len(motor_ids))
        controller.sync_write_torque_enable(motor_ids, [1] * len(motor_ids))
        time.sleep(0.2)

        wait_for_user(prompt, "Moving to calibrated zero.")
        move(controller, motors, "zero", amplitude_rad)
        time.sleep(args.hold)
        print_positions(controller, motor_ids)

        names = SEQUENCE if args.preset == "sequence" else [args.preset, "zero"]
        for name in names:
            wait_for_user(prompt, f"Moving preset: {name}.")
            move(controller, motors, name, amplitude_rad)
            time.sleep(args.hold)
            print_positions(controller, motor_ids)

    except KeyboardInterrupt:
        print("\nInterrupted. Returning to zero.")
        move(controller, motors, "zero", amplitude_rad)
        time.sleep(0.5)
    finally:
        if not args.keep_torque:
            print("Disabling torque.")
            controller.sync_write_torque_enable(motor_ids, [0] * len(motor_ids))


if __name__ == "__main__":
    main()
